import os
import time
import ast
import re
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.schemas.predict import (
    PredictionResponse,
    ScoreItem,
    RetrievedCandidateItem,
)

logger = logging.getLogger("uvicorn.error")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def clean_context_text(raw_context: Any) -> str:
    """
    Cleans raw context string (which might be a stringified python dictionary or JSON)
    into a coherent, structured medical text: 'SECTION: Sentence. SECTION: Sentence.'
    """
    if raw_context is None:
        return ""
    if not isinstance(raw_context, str):
        if isinstance(raw_context, dict):
            contexts = raw_context.get("contexts", [])
            labels = raw_context.get("labels", [])
            if contexts and labels and len(contexts) == len(labels):
                sections = [
                    f"{str(lbl).strip().upper()}: {str(txt).strip()}"
                    for lbl, txt in zip(labels, contexts)
                    if str(txt).strip()
                ]
                return " ".join(sections)
            elif contexts:
                return " ".join([str(c).strip() for c in contexts if str(c).strip()])
        return str(raw_context).strip()

    raw_context = raw_context.strip()
    if raw_context.startswith("{") and ("contexts" in raw_context or "labels" in raw_context):
        try:
            parsed = ast.literal_eval(raw_context)
            if isinstance(parsed, dict):
                contexts = parsed.get("contexts", [])
                labels = parsed.get("labels", [])
                if contexts and labels and len(contexts) == len(labels):
                    sections = [
                        f"{str(lbl).strip().upper()}: {str(txt).strip()}"
                        for lbl, txt in zip(labels, contexts)
                        if str(txt).strip()
                    ]
                    return " ".join(sections)
                elif contexts:
                    return " ".join([str(c).strip() for c in contexts if str(c).strip()])
        except Exception:
            pass

    return re.sub(r"\s+", " ", raw_context).strip()


class ClinicalQAService:
    """
    Singleton Inference Service for Open-Domain Clinical Question Answering (ODQA).
    Combines FAISS Vector Retriever with Fine-Tuned PubMedBERT Clinical Reader.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClinicalQAService, cls).__new__(cls)
            cls._instance.is_ready = False
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.retriever_model = None
            cls._instance.faiss_index = None
            cls._instance.contexts = []
            cls._instance.metadata = []
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls._instance

    def load_model(self):
        """
        Loads the Reader model, Tokenizer, Retriever Embedder, and FAISS index into memory.
        """
        logger.info(f"Initializing Clinical QA Engine on device: {self.device.upper()}...")
        start_load_time = time.perf_counter()

        # 1. Load Fine-Tuned PubMedBERT Reader
        if os.path.exists(settings.PUBMEDBERT_MODEL_PATH):
            logger.info(f"Loading PubMedBERT Reader weights from: {settings.PUBMEDBERT_MODEL_PATH}")
            self.tokenizer = AutoTokenizer.from_pretrained(settings.PUBMEDBERT_MODEL_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.PUBMEDBERT_MODEL_PATH
            )
            self.model.to(self.device)
            self.model.eval()
        else:
            logger.warning(
                f"PubMedBERT weights not found at {settings.PUBMEDBERT_MODEL_PATH}. Using baseline architecture."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                num_labels=3,
            )
            self.model.to(self.device)
            self.model.eval()

        # 2. Load Sentence Transformer Embedder
        logger.info(f"Loading Retriever Embedder: {settings.EMBEDDER_MODEL}")
        self.retriever_model = SentenceTransformer(settings.EMBEDDER_MODEL, device=self.device)

        # 3. Load FAISS Index & Context Lookups
        if os.path.exists(settings.FAISS_INDEX_PATH) and os.path.exists(settings.CONTEXTS_PATH):
            logger.info(f"Loading FAISS Vector Index from: {settings.FAISS_INDEX_PATH}")
            self.faiss_index = faiss.read_index(settings.FAISS_INDEX_PATH)
            self.contexts = np.load(settings.CONTEXTS_PATH, allow_pickle=True).tolist()
            logger.info(f"Loaded {len(self.contexts)} medical contexts in FAISS index.")
            
            if os.path.exists(settings.METADATA_PATH):
                self.metadata = np.load(settings.METADATA_PATH, allow_pickle=True).tolist()
            else:
                self.metadata = [{"context": c} for c in self.contexts]
        else:
            logger.warning("FAISS index or contexts file missing. Retriever will use empty index.")
            self.faiss_index = None
            self.contexts = []
            self.metadata = []

        # 4. Warm-up Pass for zero cold-start delay
        self._warmup()

        self.is_ready = True
        elapsed = round((time.perf_counter() - start_load_time), 2)
        logger.info(f"Clinical QA Service loaded successfully in {elapsed}s.")

    def _warmup(self):
        """Warm up the PyTorch CUDA and HuggingFace pipelines."""
        try:
            sample_q = "Is aspirin effective for acute myocardial infarction?"
            sample_ctx = "Aspirin significantly reduces mortality in patients with acute myocardial infarction."
            inputs = self.tokenizer(
                sample_q,
                sample_ctx,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
            if self.retriever_model:
                _ = self.retriever_model.encode([sample_q], convert_to_numpy=True, normalize_embeddings=True)
            logger.info("Model warm-up completed.")
        except Exception as e:
            logger.warning(f"Model warm-up skipped: {e}")

    def retrieve_evidence(self, question: str, top_k: int = 3) -> tuple[str, List[RetrievedCandidateItem], float]:
        """
        Queries FAISS vector database to retrieve top matching clinical abstracts.
        Returns:
            (primary_context, list_of_candidates, retrieval_time_ms)
        """
        start_retrieval = time.perf_counter()
        
        if self.faiss_index is None or len(self.contexts) == 0:
            return (
                "No medical knowledge base index loaded.",
                [],
                0.0,
            )

        # Generate query vector with unit normalization for Cosine similarity
        q_emb = self.retriever_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # FAISS search
        scores, indices = self.faiss_index.search(q_emb, min(top_k, len(self.contexts)))
        
        candidates: List[RetrievedCandidateItem] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.contexts):
                continue
            ctx_text = self.contexts[idx]
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            
            candidates.append(
                RetrievedCandidateItem(
                    rank=rank,
                    pubid=meta.get("pubid"),
                    question=meta.get("question"),
                    context=clean_context_text(ctx_text),
                    similarity_score=float(round(score, 4)),
                )
            )

        retrieval_time_ms = round((time.perf_counter() - start_retrieval) * 1000, 2)
        primary_context = candidates[0].context if candidates else "No relevant context found."
        
        return primary_context, candidates, retrieval_time_ms

    def predict(self, question: str, context: Optional[str] = None, top_k: int = 3) -> PredictionResponse:
        """
        Executes end-to-end Open-Domain QA (Retriever + Reader) or Direct Reading.
        """
        total_start = time.perf_counter()

        # Step A: Evidence Acquisition (Retriever vs User Context)
        if context and context.strip():
            mode = "direct_reading"
            primary_evidence = clean_context_text(context.strip())
            candidates = [
                RetrievedCandidateItem(
                    rank=1,
                    pubid=None,
                    question=None,
                    context=primary_evidence,
                    similarity_score=1.0,
                )
            ]
            retrieval_time_ms = 0.0
        else:
            mode = "retriever_reader"
            primary_evidence, candidates, retrieval_time_ms = self.retrieve_evidence(question, top_k=top_k)

        # Step B: Clinical Reader (PubMedBERT classification)
        reader_start = time.perf_counter()
        
        # Tokenize pair: [CLS] Question [SEP] Evidence [SEP]
        inputs = self.tokenizer(
            question,
            primary_evidence,
            max_length=settings.MAX_SEQUENCE_LENGTH,
            padding=True,
            truncation="only_second",  # Truncate context if exceeding 512, keeping entire question
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        reader_time_ms = round((time.perf_counter() - reader_start) * 1000, 2)
        total_time_ms = round((time.perf_counter() - total_start) * 1000, 2)

        # Mapping: 0 -> No, 1 -> Yes, 2 -> Maybe
        labels = ["No", "Yes", "Maybe"]
        scores_list = [
            ScoreItem(label=labels[i], score=float(round(probs[i], 4)))
            for i in range(len(labels))
        ]

        # Highest probability decision
        pred_idx = int(np.argmax(probs))
        predicted_label = labels[pred_idx]
        confidence = float(round(probs[pred_idx], 4))

        return PredictionResponse(
            task="Clinical Question Answering",
            question=question,
            prediction=predicted_label,
            confidence=confidence,
            scores=scores_list,
            retrieved_context=primary_evidence,
            candidates=candidates,
            mode=mode,
            retrieval_time_ms=retrieval_time_ms,
            inference_time_ms=reader_time_ms,
            total_time_ms=total_time_ms,
            device=self.device,
            model_name="PubMedBERT Fine-Tuned (3-class)",
        )


model_service = ClinicalQAService()
