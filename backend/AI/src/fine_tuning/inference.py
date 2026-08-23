"""
inference.py
============
Inference module for the fine-tuned Clinical QA BioBERT model.

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

import torch
import torch.nn.functional as F
import yaml
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[4]   # …/Clinical-QA-NLP
AI_DIR       = PROJECT_ROOT / "backend" / "AI"
CONFIG_PATH  = AI_DIR / "config" / "inference_config.yaml"


# ── Config loader ─────────────────────────────────────────────────────────────
def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Inference config not found:\n  {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Inference class ───────────────────────────────────────────────────────────
class BioBERTInference:
    """
    Loads the fine-tuned BioBERT checkpoint and provides predict() for
    single-sample and batch inference.
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        cfg = load_config(config_path)

        # ── Device ────────────────────────────────────────────────────────────
        device_cfg = cfg["inference"]["device"]
        if device_cfg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_cfg)

        # ── Model + tokenizer ─────────────────────────────────────────────────
        checkpoint_dir = str(AI_DIR / cfg["model"]["checkpoint_dir"])
        print(f"Loading model from: {checkpoint_dir}")
        print(f"Device: {self.device}\n")

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir,
            num_labels=cfg["model"]["num_labels"],
        )
        self.model.to(self.device)
        self.model.eval()

        # ── Config values ─────────────────────────────────────────────────────
        self.max_len    = cfg["model"]["max_len"]
        self.threshold  = cfg["inference"]["confidence_threshold"]
        # Convert YAML int keys {"0": "No", "1": "Yes", "2": "Maybe"} to int
        self.id2label   = {int(k): v for k, v in cfg["labels"].items()}

    # ── Single prediction ─────────────────────────────────────────────────────
    def predict(self, question: str, context: str) -> dict:
        """
        Classify a single (question, context) pair.

        Args:
            question: The biomedical yes/no question.
            context:  Supporting evidence text (abstract sentences).

        Returns:
            {
              "label":      "Yes" | "No" | "Maybe",
              "confidence": float,   # probability of the predicted class
              "scores":     {"Yes": float, "No": float, "Maybe": float}
            }
        """
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs      = F.softmax(logits, dim=1).squeeze().cpu().tolist()
        pred_idx   = int(torch.argmax(logits, dim=1).item())
        confidence = probs[pred_idx]
        label      = self.id2label[pred_idx]

        scores = {self.id2label[i]: round(probs[i], 4) for i in range(len(probs))}

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "scores":     scores,
        }

    # ── Batch inference ───────────────────────────────────────────────────────
    def predict_batch(self, questions: list[str], contexts: list[str]) -> list[dict]:
        """
        Classify a batch of (question, context) pairs efficiently.

        Args:
            questions: List of question strings.
            contexts:  List of context strings (same length as questions).

        Returns:
            List of result dicts (same format as predict()).
        """
        encoding = self.tokenizer(
            questions,
            contexts,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs    = F.softmax(logits, dim=1).cpu().tolist()
        pred_ids = torch.argmax(logits, dim=1).cpu().tolist()

        results = []
        for i, pred_idx in enumerate(pred_ids):
            confidence = probs[i][pred_idx]
            label      = self.id2label[pred_idx]
            scores     = {self.id2label[j]: round(probs[i][j], 4) for j in range(len(probs[i]))}
            results.append({
                "label":      label,
                "confidence": round(confidence, 4),
                "scores":     scores,
            })
        return results

    # ── CSV batch runner ──────────────────────────────────────────────────────
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

        all_results: list[dict] = []

        for start in tqdm(range(0, len(df), batch_size), desc="Inferring"):
            batch_df   = df.iloc[start : start + batch_size]
            questions  = batch_df["question"].fillna("").astype(str).tolist()
            contexts   = batch_df["context"].fillna("").astype(str).tolist()
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


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()

    model = BioBERTInference()

    # Batch mode: score the configured input CSV
    input_csv  = AI_DIR / cfg["batch"]["input_csv"]
    output_csv = AI_DIR / cfg["batch"]["output_csv"]
    batch_size = cfg["inference"]["batch_size"]

    model.run_on_csv(input_csv, output_csv, batch_size=batch_size)
