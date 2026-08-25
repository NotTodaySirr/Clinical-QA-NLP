"""
inference.py
============
Inference module for fine-tuned Clinical QA Transformer models (PubMedBERT / BioBERT).

Supports two modes
------------------
  1. Single prediction  — call predict() with a question + context string
  2. Batch CSV inference — run this file directly to score a full CSV

Config
------
  All settings are loaded from:
    backend/AI/config/inference_config.yaml

Usage
-----
  # Single prediction (import as module):
      from fine_tuning.inference import BioBERTInference
      model = BioBERTInference()
      result = model.predict(question="...", context="...")
      print(result)   # {"label": "Yes", "confidence": 0.92, "scores": {...}}

  # Batch CSV scoring (run directly):
      python backend/AI/src/fine_tuning/inference.py
"""

import sys
from pathlib import Path
from typing import Any, List, Dict, Optional

import torch
import torch.nn.functional as F
import yaml
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path setup 
PROJECT_ROOT = Path(__file__).resolve().parents[4]   # …/Clinical-QA-NLP
AI_DIR       = PROJECT_ROOT / "backend" / "AI"
SRC_DIR      = AI_DIR / "src"
CONFIG_PATH  = AI_DIR / "config" / "inference_config.yaml"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_process.data_ingestion import clean_context_text  


# Config loader 
def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Inference config not found:\n  {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Inference class 
class BioBERTInference:
    """
    Loads the fine-tuned checkpoint and provides predict() for
    single-sample and batch inference with optional probability threshold calibration.
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        cfg = load_config(config_path)

        # Device 
        device_cfg = cfg["inference"]["device"]
        if device_cfg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_cfg)

        #  Model + tokenizer path resolution (with fallback) 
        primary_dir = AI_DIR / cfg["model"]["checkpoint_dir"]
        fallback_dir = AI_DIR / "saved_model" / "biobert_finetuned"

        if primary_dir.exists():
            checkpoint_dir = str(primary_dir)
        elif fallback_dir.exists():
            print(f"Warning: Primary checkpoint {primary_dir} not found. Falling back to {fallback_dir}")
            checkpoint_dir = str(fallback_dir)
        else:
            checkpoint_dir = str(primary_dir)

        print(f"Loading model checkpoint from: {checkpoint_dir}")
        print(f"Device: {self.device}\n")

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir,
            num_labels=cfg["model"]["num_labels"],
        )
        self.model.to(self.device)
        self.model.eval()

        # Config values 
        self.max_len         = cfg["model"]["max_len"]
        self.confidence_thr  = cfg["inference"].get("confidence_threshold", 0.0)
        self.maybe_threshold = float(cfg["inference"].get("maybe_threshold", 0.0))
        # Convert YAML int keys {"0": "No", "1": "Yes", "2": "Maybe"} to int
        self.id2label        = {int(k): v for k, v in cfg["labels"].items()}

    # Single prediction 
    def predict(self, question: str, context: Any) -> dict:
        """
        Classify a single (question, context) pair.

        Args:
            question: The biomedical yes/no question.
            context:  Supporting evidence text (raw string, dict, or structured text).

        Returns:
            {
              "label":      "Yes" | "No" | "Maybe",
              "confidence": float,   # probability of the predicted class
              "scores":     {"Yes": float, "No": float, "Maybe": float}
            }
        """
        cleaned_context = clean_context_text(context)

        encoding = self.tokenizer(
            str(question).strip(),
            cleaned_context,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()
        if not isinstance(probs, list):
            probs = [probs]

        # Decision rule: Calibrated threshold for Maybe vs standard argmax
        if self.maybe_threshold > 0.0 and len(probs) >= 3:
            if probs[2] >= self.maybe_threshold:
                pred_idx = 2
            else:
                pred_idx = 1 if probs[1] > probs[0] else 0
        else:
            pred_idx = int(torch.argmax(logits, dim=1).item())

        confidence = probs[pred_idx]
        label      = self.id2label[pred_idx]

        scores = {self.id2label[i]: round(probs[i], 4) for i in range(len(probs))}

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "scores":     scores,
        }

    # Batch inference 
    def predict_batch(self, questions: List[str], contexts: List[Any]) -> List[dict]:
        """
        Classify a batch of (question, context) pairs efficiently.

        Args:
            questions: List of question strings.
            contexts:  List of context strings/dicts (same length as questions).

        Returns:
            List of result dicts (same format as predict()).
        """
        cleaned_questions = [str(q).strip() for q in questions]
        cleaned_contexts  = [clean_context_text(c) for c in contexts]

        encoding = self.tokenizer(
            cleaned_questions,
            cleaned_contexts,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs = F.softmax(logits, dim=1).cpu().tolist()

        results = []
        for i in range(len(probs)):
            p = probs[i]
            if self.maybe_threshold > 0.0 and len(p) >= 3:
                if p[2] >= self.maybe_threshold:
                    pred_idx = 2
                else:
                    pred_idx = 1 if p[1] > p[0] else 0
            else:
                pred_idx = int(torch.argmax(logits[i], dim=-1).item())

            confidence = p[pred_idx]
            label      = self.id2label[pred_idx]
            scores     = {self.id2label[j]: round(p[j], 4) for j in range(len(p))}
            results.append({
                "label":      label,
                "confidence": round(confidence, 4),
                "scores":     scores,
            })
        return results

    # CSV batch runner 
    def run_on_csv(self, input_csv: Path, output_csv: Path, batch_size: int = 16) -> None:
        """
        Run inference on every row of a CSV file and save results.

        Expected CSV columns: question, context
        Output adds:          predicted_label, confidence, score_yes, score_no, score_maybe
        """
        print(f"Running batch inference on: {input_csv}")
        df = pd.read_csv(input_csv)

        required = {"question", "context"}
        if not required.issubset(df.columns):
            raise ValueError(f"Input CSV must contain columns: {required}")

        all_results: List[dict] = []

        for start in tqdm(range(0, len(df), batch_size), desc="Inferring"):
            batch_df   = df.iloc[start : start + batch_size]
            questions  = batch_df["question"].fillna("").tolist()
            contexts   = batch_df["context"].fillna("").tolist()
            all_results.extend(self.predict_batch(questions, contexts))

        results_df = pd.DataFrame(all_results)
        df["predicted_label"] = results_df["label"].values
        df["confidence"]      = results_df["confidence"].values
        df["score_yes"]       = results_df["scores"].apply(lambda s: s.get("Yes", 0.0)).values
        df["score_no"]        = results_df["scores"].apply(lambda s: s.get("No",  0.0)).values
        df["score_maybe"]     = results_df["scores"].apply(lambda s: s.get("Maybe", 0.0)).values

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")

        # Summary
        dist = df["predicted_label"].value_counts()
        print("\nPrediction distribution:")
        for label, count in dist.items():
            pct = count / len(df) * 100
            print(f"  {label:<8}: {count:>6,}  ({pct:.1f}%)")


# Alias for backward compatibility
PubMedBERTInference = BioBERTInference


# CLI entry point 
if __name__ == "__main__":
    cfg = load_config()

    model = BioBERTInference()

    # Batch mode: score the configured input CSV
    input_csv  = AI_DIR / cfg["batch"]["input_csv"]
    output_csv = AI_DIR / cfg["batch"]["output_csv"]
    batch_size = cfg["inference"]["batch_size"]

    model.run_on_csv(input_csv, output_csv, batch_size=batch_size)

