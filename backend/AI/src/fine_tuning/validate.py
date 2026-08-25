"""
validate.py
===========
Unified Validation and Benchmark Evaluation for Clinical QA Models.

Features:
  1. Evaluates on BOTH:
     - Held-out Test Split (1,050 samples from pubmedqa_labeled.csv)
     - Official 1,000-sample Gold Standard Ground Truth (pubmedqa_labeled_gold_standard.csv)
  2. Supports evaluating BioBERT, PubMedBERT, or both models simultaneously (--model all).
  3. Uses clean abstract parsing to prevent truncation of clinical results.
  4. Exports structured JSON metrics and CSV predictions into model-specific directories.
  5. Computes multi-class metrics, confusion matrices, severe inversions (Yes <-> No), and confidence analysis.

Usage:
  # Evaluate default model (PubMedBERT)
  uv run python backend/AI/src/fine_tuning/validate.py

  # Evaluate both models and generate a side-by-side comparison report
  uv run python backend/AI/src/fine_tuning/validate.py --model all

  # Evaluate a specific model
  uv run python backend/AI/src/fine_tuning/validate.py --model biobert
  uv run python backend/AI/src/fine_tuning/validate.py --model pubmedbert

  # Evaluate with calibrated Maybe threshold
  uv run python backend/AI/src/fine_tuning/validate.py --model pubmedbert --maybe-threshold 0.28
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Safe UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[4]
AI_DIR = PROJECT_ROOT / "backend" / "AI"
SRC_DIR = AI_DIR / "src"
CONFIG_PATH = AI_DIR / "config" / "validation_config.yaml"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from data_process.data_ingestion import clean_context_text

REV_LABEL_MAP = {
    "No": 0, "no": 0, 0: 0,
    "Yes": 1, "yes": 1, 1: 1,
    "Maybe": 2, "maybe": 2, 2: 2
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Validation config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _header(title: str) -> None:
    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")


# ── PyTorch Dataset Wrapper ───────────────────────────────────────────────────
class EvaluationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = str(row.get("question", "")).strip()
        raw_context = row.get("context", "")
        cleaned_context = clean_context_text(raw_context)
        label = int(row["label_id"])

        encoding = self.tokenizer(
            question,
            cleaned_context,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ── Data Loaders ──────────────────────────────────────────────────────────────
def get_held_out_test_df(cfg: dict) -> pd.DataFrame:
    labeled_csv = AI_DIR / cfg["data"]["labeled_csv"]
    if not labeled_csv.exists():
        raise FileNotFoundError(f"Labeled CSV not found: {labeled_csv}")

    df = pd.read_csv(labeled_csv)
    df["label_id"] = df["label_decision"].map(REV_LABEL_MAP)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    _, temp_df = train_test_split(
        df,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=df["label_id"],
    )
    _, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=cfg["data"]["random_state"],
        stratify=temp_df["label_id"],
    )
    return test_df.reset_index(drop=True)


def get_gold_standard_df(cfg: dict) -> pd.DataFrame:
    gold_csv = AI_DIR / cfg["data"]["gold_standard_csv"]
    if not gold_csv.exists():
        raise FileNotFoundError(f"Gold Standard CSV not found: {gold_csv}")

    df = pd.read_csv(gold_csv)
    dec_col = "final_decision" if "final_decision" in df.columns else "label_decision"
    df["label_id"] = df[dec_col].map(REV_LABEL_MAP)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)
    return df.reset_index(drop=True)


# ── Core Inference Routine ────────────────────────────────────────────────────
def run_dataset_inference(
    model, tokenizer, df: pd.DataFrame, device, batch_size: int = 16, max_len: int = 512, maybe_threshold: float = 0.0
) -> Tuple[List[int], List[int], List[float], np.ndarray]:
    dataset = EvaluationDataset(df, tokenizer, max_length=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred, confidences, all_probs = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

            if maybe_threshold > 0.0:
                batch_preds = []
                batch_confs = []
                for p in probs:
                    if p[2] >= maybe_threshold:
                        batch_preds.append(2)
                        batch_confs.append(float(p[2]))
                    else:
                        pred_idx = 1 if p[1] > p[0] else 0
                        batch_preds.append(pred_idx)
                        batch_confs.append(float(p[pred_idx]))
            else:
                batch_preds = np.argmax(probs, axis=1).tolist()
                batch_confs = probs[np.arange(len(batch_preds)), batch_preds].tolist()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(batch_preds)
            confidences.extend(batch_confs)
            all_probs.extend(probs.tolist())

    return y_true, y_pred, confidences, np.array(all_probs)


# ── Metric Computation ────────────────────────────────────────────────────────
def calculate_metrics(
    y_true: List[int], y_pred: List[int], confidences: List[float], label_names: List[str], conf_threshold: float = 0.70
) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))

    clf_report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = (cm.astype(float) / cm.sum(axis=1, keepdims=True)).tolist()

    # Severe inversions: True No predicted Yes (cm[0][1]) + True Yes predicted No (cm[1][0])
    severe_flips = int(cm[0][1] + cm[1][0])
    severe_pct = round((severe_flips / len(y_true)) * 100, 2)

    confs_arr = np.array(confidences)
    n_uncertain = int((confs_arr < conf_threshold).sum())

    return {
        "overall_accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "weighted_f1": round(weighted_f1 * 100, 2),
        "micro_f1": round(micro_f1 * 100, 2),
        "severe_inversions_count": severe_flips,
        "severe_inversions_pct": severe_pct,
        "classification_report": clf_report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalised": cm_norm,
        "confidence": {
            "mean": round(float(confs_arr.mean()), 4),
            "median": round(float(np.median(confs_arr)), 4),
            "std": round(float(confs_arr.std()), 4),
            "pct_uncertain": round((n_uncertain / len(confs_arr)) * 100, 2),
        },
    }


def print_metric_summary(title: str, metrics: Dict[str, Any], label_names: List[str]):
    print(f"\n[{title}]")
    print(f"  Accuracy            : {metrics['overall_accuracy']:.2f}%")
    print(f"  Macro F1            : {metrics['macro_f1']:.2f}%")
    print(f"  Weighted F1         : {metrics['weighted_f1']:.2f}%")
    print(f"  Severe Inversions   : {metrics['severe_inversions_count']} ({metrics['severe_inversions_pct']}%)")
    print(f"  Mean Confidence     : {metrics['confidence']['mean']:.4f} (Uncertain < 0.70: {metrics['confidence']['pct_uncertain']}%)")

    print("\n  Per-Class Performance:")
    for lbl in label_names:
        stats = metrics["classification_report"].get(lbl, {})
        p, r, f = stats.get("precision", 0) * 100, stats.get("recall", 0) * 100, stats.get("f1-score", 0) * 100
        sup = stats.get("support", 0)
        print(f"    {lbl:<8}: Precision {p:>5.1f}% | Recall {r:>5.1f}% | F1 {f:>5.1f}% | Support: {sup:>4}")

    cm = np.array(metrics["confusion_matrix"])
    print("\n  Confusion Matrix (Row = True, Col = Predicted):")
    print(f"            {label_names[0]:>10} {label_names[1]:>10} {label_names[2]:>10}")
    for i, lbl in enumerate(label_names):
        print(f"  {lbl:>8}: {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")


# ── Model Evaluator & Exporter ────────────────────────────────────────────────
def evaluate_single_model(
    model_name: str,
    checkpoint_dir: Path,
    cfg: dict,
    held_out_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    device,
    maybe_threshold: float = 0.0,
) -> Dict[str, Any]:
    _header(f"Evaluating Model: {model_name}")
    print(f"Checkpoint Directory : {checkpoint_dir}")
    print(f"Device               : {device}")
    print(f"Threshold Calibration: {'Disabled (argmax)' if maybe_threshold <= 0 else f'Maybe >= {maybe_threshold}'}")

    if not checkpoint_dir.exists():
        print(f"[ERROR] Checkpoint directory does not exist: {checkpoint_dir}")
        return {}

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    model.to(device)
    model.eval()

    label_names = ["No", "Yes", "Maybe"]
    batch_size = cfg["evaluation"]["batch_size"]
    max_len = cfg["evaluation"]["max_len"]
    conf_thresh = cfg["evaluation"]["confidence_threshold"]

    # 1. Evaluate Held-out Test Split
    print("\nRunning inference on Held-out Test Split (1,050 samples)...")
    y_true_test, y_pred_test, confs_test, probs_test = run_dataset_inference(
        model, tokenizer, held_out_df, device, batch_size, max_len, maybe_threshold
    )
    test_metrics = calculate_metrics(y_true_test, y_pred_test, confs_test, label_names, conf_thresh)
    print_metric_summary("Held-out Test Split Results", test_metrics, label_names)

    # 2. Evaluate Gold Standard Benchmark
    print("\nRunning inference on Official 1,000 Gold Standard (Reasoning-Required)...")
    y_true_gold, y_pred_gold, confs_gold, probs_gold = run_dataset_inference(
        model, tokenizer, gold_df, device, batch_size, max_len, maybe_threshold
    )
    gold_metrics = calculate_metrics(y_true_gold, y_pred_gold, confs_gold, label_names, conf_thresh)
    print_metric_summary("Official Gold Standard Benchmark Results", gold_metrics, label_names)

    # 3. Export Artifacts
    output_dir = AI_DIR / cfg["output"]["base_dir"] / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg["output"]["save_json"]:
        with open(output_dir / "held_out_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        with open(output_dir / "gold_standard_metrics.json", "w", encoding="utf-8") as f:
            json.dump(gold_metrics, f, indent=2)

        summary = {
            "model_name": model_name,
            "checkpoint_dir": str(checkpoint_dir),
            "held_out_test": {
                "accuracy": test_metrics["overall_accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "weighted_f1": test_metrics["weighted_f1"],
                "severe_inversions_pct": test_metrics["severe_inversions_pct"],
            },
            "gold_standard": {
                "accuracy": gold_metrics["overall_accuracy"],
                "macro_f1": gold_metrics["macro_f1"],
                "weighted_f1": gold_metrics["weighted_f1"],
                "severe_inversions_pct": gold_metrics["severe_inversions_pct"],
            },
        }
        with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  JSON Metrics saved to -> {output_dir}")

    if cfg["output"]["save_csv"]:
        # Save Held-out Predictions
        pred_test_df = held_out_df.copy()
        pred_test_df["true_label"] = [label_names[i] for i in y_true_test]
        pred_test_df["predicted_label"] = [label_names[i] for i in y_pred_test]
        pred_test_df["confidence"] = [round(c, 4) for c in confs_test]
        pred_test_df["score_no"] = [round(p[0], 4) for p in probs_test]
        pred_test_df["score_yes"] = [round(p[1], 4) for p in probs_test]
        pred_test_df["score_maybe"] = [round(p[2], 4) for p in probs_test]
        pred_test_df["correct"] = [t == p for t, p in zip(y_true_test, y_pred_test)]
        pred_test_df.to_csv(output_dir / "held_out_test_predictions.csv", index=False)

        # Save Gold Predictions
        pred_gold_df = gold_df.copy()
        pred_gold_df["true_label"] = [label_names[i] for i in y_true_gold]
        pred_gold_df["predicted_label"] = [label_names[i] for i in y_pred_gold]
        pred_gold_df["confidence"] = [round(c, 4) for c in confs_gold]
        pred_gold_df["score_no"] = [round(p[0], 4) for p in probs_gold]
        pred_gold_df["score_yes"] = [round(p[1], 4) for p in probs_gold]
        pred_gold_df["score_maybe"] = [round(p[2], 4) for p in probs_gold]
        pred_gold_df["correct"] = [t == p for t, p in zip(y_true_gold, y_pred_gold)]
        pred_gold_df.to_csv(output_dir / "gold_standard_predictions.csv", index=False)
        print(f"  CSV Predictions saved to -> {output_dir}")

    return {
        "test": test_metrics,
        "gold": gold_metrics,
    }


# ── Main Entry Point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Unified Clinical QA Model Validation & Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="pubmedbert",
        choices=["biobert", "pubmedbert", "all"],
        help="Which model checkpoint to evaluate ('biobert', 'pubmedbert', or 'all')",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Optional custom checkpoint path. Overrides --model choice.",
    )
    parser.add_argument(
        "--maybe-threshold",
        type=float,
        default=0.0,
        help="Optional probability threshold for Maybe class (e.g. 0.28). Default 0.0 uses argmax.",
    )
    args = parser.parse_args()

    cfg = load_config()
    device_cfg = cfg["evaluation"]["device"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_cfg == "auto" else device_cfg
    )

    print("\n" + "=" * 75)
    print("  CLINICAL QA UNIFIED VALIDATION & BENCHMARK SUITE")
    print("=" * 75)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    held_out_df = get_held_out_test_df(cfg)
    gold_df = get_gold_standard_df(cfg)

    results = {}

    if args.model_dir:
        custom_dir = Path(args.model_dir)
        mname = custom_dir.name
        results[mname] = evaluate_single_model(
            mname, custom_dir, cfg, held_out_df, gold_df, device, args.maybe_threshold
        )
    elif args.model == "all":
        for mkey in ["biobert", "pubmedbert"]:
            m_path = AI_DIR / cfg["models"][mkey]
            mname = f"{mkey}_finetuned"
            if m_path.exists():
                results[mname] = evaluate_single_model(
                    mname, m_path, cfg, held_out_df, gold_df, device, args.maybe_threshold
                )
            else:
                print(f"[WARN] Checkpoint for '{mkey}' not found at: {m_path}")
    else:
        mkey = args.model
        m_path = AI_DIR / cfg["models"][mkey]
        mname = f"{mkey}_finetuned"
        results[mname] = evaluate_single_model(
            mname, m_path, cfg, held_out_df, gold_df, device, args.maybe_threshold
        )

    # If multiple models were evaluated, print and save a comparative table
    if len(results) > 1:
        _header("MODEL COMPARISON SUMMARY")
        print(f"{'Model Name':<24} | {'Gold Acc':<10} | {'Gold Macro F1':<14} | {'Gold Severe Flips':<18} | {'Test Acc':<10} | {'Test Macro F1':<14}")
        print("-" * 105)
        comparison_dict = {}
        for mname, res in results.items():
            if not res:
                continue
            g_acc = f"{res['gold']['overall_accuracy']:.2f}%"
            g_f1 = f"{res['gold']['macro_f1']:.2f}%"
            g_flips = f"{res['gold']['severe_inversions_pct']:.2f}%"
            t_acc = f"{res['test']['overall_accuracy']:.2f}%"
            t_f1 = f"{res['test']['macro_f1']:.2f}%"
            print(f"{mname:<24} | {g_acc:<10} | {g_f1:<14} | {g_flips:<18} | {t_acc:<10} | {t_f1:<14}")
            comparison_dict[mname] = {
                "gold_accuracy": res["gold"]["overall_accuracy"],
                "gold_macro_f1": res["gold"]["macro_f1"],
                "gold_severe_inversions_pct": res["gold"]["severe_inversions_pct"],
                "test_accuracy": res["test"]["overall_accuracy"],
                "test_macro_f1": res["test"]["macro_f1"],
            }
        print("=" * 105)

        comp_path = AI_DIR / cfg["output"]["base_dir"] / "models_comparison_report.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison_dict, f, indent=2)
        print(f"\n  Cross-model comparison report saved to -> {comp_path}\n")


if __name__ == "__main__":
    main()
