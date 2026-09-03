"""
FAISS Vector Knowledge Base Generator for Clinical QA
Builds an optimized FAISS vector index from PubMedQA labeled abstracts.
"""

import os
import sys
import time
import ast
import re
import argparse
from typing import Any
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


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


def build_faiss_index(
    csv_path: str = "backend/AI/data/labeled/pubmedqa_labeled.csv",
    output_dir: str = "backend/AI/saved_model",
    embedder_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
    batch_size: int = 256,
):
    print("=" * 60)
    print("[FAISS] Starting Knowledge Base Index Generation (with cleaned text)")
    print("=" * 60)
    start_time = time.time()

    if not os.path.exists(csv_path):
        # Auto-detect path if executed from backend/ instead of root or vice versa
        candidates = [
            csv_path.replace("backend/", ""),
            os.path.join("backend", csv_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "labeled", "pubmedqa_labeled.csv"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                csv_path = candidate
                break
        else:
            raise FileNotFoundError(f"Source dataset not found at: {csv_path}")

    if not os.path.exists(output_dir):
        alt_out = output_dir.replace("backend/", "") if output_dir.startswith("backend/") else os.path.join("backend", output_dir)
        if os.path.exists(os.path.dirname(alt_out)) or os.path.exists(alt_out):
            output_dir = alt_out

    os.makedirs(output_dir, exist_ok=True)
    index_output_path = os.path.join(output_dir, "faiss_medical.index")
    contexts_output_path = os.path.join(output_dir, "contexts.npy")
    metadata_output_path = os.path.join(output_dir, "contexts_meta.npy")

    print(f"[FAISS] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[FAISS] Loaded {len(df)} rows.")

    # Clean context text
    df["clean_context"] = df["context"].apply(clean_context_text)
    df_clean = df.dropna(subset=["clean_context"]).drop_duplicates(subset=["clean_context"]).reset_index(drop=True)
    contexts = df_clean["clean_context"].astype(str).tolist()
    metadata = df_clean[["pubid", "question", "clean_context", "label_decision"]].rename(
        columns={"clean_context": "context"}
    ).to_dict(orient="records")
    print(f"[FAISS] Extracted {len(contexts)} unique cleaned medical abstracts.")

    print(f"[FAISS] Loading SentenceTransformer embedding model: {embedder_model}")
    embedder = SentenceTransformer(embedder_model)

    print(f"[FAISS] Encoding {len(contexts)} abstracts in batches of {batch_size}...")
    embeddings = embedder.encode(
        contexts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Unit length for cosine similarity via Inner Product
    )
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    print(f"[FAISS] Embeddings shape: {embeddings.shape} (Dimension: {dim})")

    print("[FAISS] Constructing FAISS IndexFlatIP (Cosine Similarity)...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[FAISS] Total vectors indexed: {index.ntotal}")

    print(f"[FAISS] Saving FAISS index to: {index_output_path}")
    faiss.write_index(index, index_output_path)

    print(f"[FAISS] Saving contexts text lookup to: {contexts_output_path}")
    np.save(contexts_output_path, np.array(contexts, dtype=object))

    print(f"[FAISS] Saving metadata lookup to: {metadata_output_path}")
    np.save(metadata_output_path, np.array(metadata, dtype=object))

    index_size_mb = os.path.getsize(index_output_path) / (1024 * 1024)
    contexts_size_mb = os.path.getsize(contexts_output_path) / (1024 * 1024)
    elapsed = round(time.time() - start_time, 2)

    print("=" * 60)
    print(f"[FAISS] SUCCESS: Knowledge Base generated in {elapsed}s")
    print(f"[FAISS] Index Size: {index_size_mb:.2f} MB | Contexts Size: {contexts_size_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS Knowledge Base for PubMedQA")
    parser.add_argument("--csv_path", type=str, default="backend/AI/data/labeled/pubmedqa_labeled.csv")
    parser.add_argument("--output_dir", type=str, default="backend/AI/saved_model")
    parser.add_argument("--embedder", type=str, default="pritamdeka/S-PubMedBert-MS-MARCO")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    build_faiss_index(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        embedder_model=args.embedder,
        batch_size=args.batch_size,
    )
