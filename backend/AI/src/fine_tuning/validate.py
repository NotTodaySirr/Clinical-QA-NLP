"""
validate.py
===========
Detailed evaluation of the fine-tuned Clinical QA BioBERT model
on the held-out test split.

Metrics reported
----------------
  - Overall accuracy
  - Per-class Precision, Recall, F1
  - Macro / Weighted / Micro F1
  - Confusion matrix (counts + normalised %)
  - Confidence distribution per predicted class
  - Uncertain predictions (below confidence threshold)
  - Misclassification analysis with sample rows

Config
------
  All settings loaded from:
    backend/AI/config/validation_config.yaml

Usage
-----
  python backend/AI/src/fine_tuning/validate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[4]
AI_DIR       = PROJECT_ROOT / "backend" / "AI"
SRC_DIR      = AI_DIR / "src"
CONFIG_PATH  = AI_DIR / "config" / "validation_config.yaml"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ── Config loader ─────────────────────────────────────────────────────────────
def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Validation config not found:\n  {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Helper: print section header ─────────────────────────────────────────────
def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Load test split from labeled CSV ─────────────────────────────────────────
def load_test_split(cfg: dict) -> pd.DataFrame:
    """
    Re-creates the identical test split used during training by applying the
    same random_state and test_size from the config.
    """
    _header("Loading Test Split")
    labeled_csv = AI_DIR / cfg["data"]["labeled_csv"]
    if not labeled_csv.exists():
        raise FileNotFoundError(f"Labeled CSV not found:\n  {labeled_csv}")

    df = pd.read_csv(labeled_csv)

    # Apply same label mapping as training
    label2id = {"No": 0, "Yes": 1, "Maybe": 2}
    df["label_id"] = df["label_decision"].map(label2id)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    # Re-create the exact same stratified split
    _, test_df = train_test_split(
        df,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=df["label_id"],
    )
    # The 20% temp set is further split 50/50 into val+test in DataProcessor
    _, test_df = train_test_split(
        test_df,
        test_size=0.5,
        random_state=cfg["data"]["random_state"],
        stratify=test_df["label_id"],
    )

    print(f"Test set: {len(test_df):,} rows")
    print(f"Label distribution:")
    id2label = {int(k): v for k, v in cfg["labels"].items()}
    for lid, count in test_df["label_id"].value_counts().sort_index().items():
        print(f"  {id2label[lid]:<8}: {count:>5,}  ({count/len(test_df)*100:.1f}%)")

    return test_df.reset_index(drop=True)


# ── Run inference on test split ───────────────────────────────────────────────
def run_inference(cfg: dict, test_df: pd.DataFrame) -> tuple[list, list, list]:
    """
    Returns (true_labels, predicted_labels, confidences).
    """
    _header("Running Inference")

    # Device
    device_cfg = cfg["evaluation"]["device"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_cfg == "auto" else device_cfg
    )
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint_dir = str(AI_DIR / cfg["model"]["checkpoint_dir"])
    print(f"Checkpoint: {checkpoint_dir}\n")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        num_labels=cfg["model"]["num_labels"],
    )
    model.to(device)
    model.eval()

    batch_size = cfg["evaluation"]["batch_size"]
    max_len    = cfg["model"]["max_len"]

    y_true, y_pred, confidences = [], [], []

    for start in tqdm(range(0, len(test_df), batch_size), desc="Evaluating"):
        batch  = test_df.iloc[start : start + batch_size]
        questions = batch["question"].fillna("").astype(str).tolist()
        contexts  = batch["context"].fillna("").astype(str).tolist()
        labels    = batch["label_id"].tolist()

        encoding = tokenizer(
            questions,
            contexts,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs    = F.softmax(logits, dim=1).cpu().numpy()
        preds    = np.argmax(probs, axis=1).tolist()
        confs    = probs[np.arange(len(preds)), preds].tolist()

        y_true.extend(labels)
        y_pred.extend(preds)
        confidences.extend(confs)

    return y_true, y_pred, confidences


# ── Compute and print all metrics ─────────────────────────────────────────────
def compute_metrics(
    cfg: dict,
    test_df: pd.DataFrame,
    y_true: list,
    y_pred: list,
    confidences: list,
) -> dict:
    id2label    = {int(k): v for k, v in cfg["labels"].items()}
    label_names = [id2label[i] for i in sorted(id2label)]
    threshold   = cfg["evaluation"]["confidence_threshold"]
    f1_avgs     = cfg["evaluation"]["f1_averaging"]

    metrics: dict = {}

    # ── 1. Overall accuracy ──────────────────────────────────────────────────
    _header("1. Overall Accuracy")
    acc = accuracy_score(y_true, y_pred)
    metrics["overall_accuracy"] = round(acc, 4)
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

    # ── 2. Per-class Precision / Recall / F1 ─────────────────────────────────
    _header("2. Per-Class Precision / Recall / F1")
    report_str = classification_report(
        y_true, y_pred,
        target_names=label_names,
        digits=4,
    )
    print(report_str)
    report_dict = classification_report(
        y_true, y_pred,
        target_names=label_names,
        output_dict=True,
    )
    metrics["classification_report"] = report_dict

    # ── 3. F1 by averaging strategy ──────────────────────────────────────────
    _header("3. F1 by Averaging Strategy")
    f1_results = {}
    for avg in f1_avgs:
        val = f1_score(y_true, y_pred, average=avg)
        f1_results[avg] = round(val, 4)
        print(f"  F1 ({avg:<10}): {val:.4f}")
    metrics["f1_scores"] = f1_results

    # ── 4. Confusion matrix ───────────────────────────────────────────────────
    _header("4. Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Print raw counts
    header = f"{'':>10}" + "".join(f"{n:>10}" for n in label_names)
    print(f"\n  Raw counts (row=True, col=Predicted):")
    print(f"  {header}")
    for i, row in enumerate(cm):
        vals = "".join(f"{v:>10}" for v in row)
        print(f"  {label_names[i]:>10}{vals}")

    # Print normalised %
    print(f"\n  Normalised % (row=True, col=Predicted):")
    print(f"  {header}")
    for i, row in enumerate(cm_norm):
        vals = "".join(f"{v*100:>9.1f}%" for v in row)
        print(f"  {label_names[i]:>10}{vals}")

    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_normalised"] = cm_norm.tolist()

    # ── 5. Confidence distribution ────────────────────────────────────────────
    _header("5. Confidence Distribution")
    confs_arr = np.array(confidences)
    print(f"  Mean confidence    : {confs_arr.mean():.4f}")
    print(f"  Median confidence  : {np.median(confs_arr):.4f}")
    print(f"  Std confidence     : {confs_arr.std():.4f}")
    print(f"  Min confidence     : {confs_arr.min():.4f}")
    print(f"  Max confidence     : {confs_arr.max():.4f}")

    n_uncertain = int((confs_arr < threshold).sum())
    pct_uncertain = n_uncertain / len(confs_arr) * 100
    print(f"\n  Below threshold ({threshold:.2f}): {n_uncertain:,}  ({pct_uncertain:.1f}%)")

    metrics["confidence"] = {
        "mean":       round(float(confs_arr.mean()), 4),
        "median":     round(float(np.median(confs_arr)), 4),
        "std":        round(float(confs_arr.std()), 4),
        "min":        round(float(confs_arr.min()), 4),
        "max":        round(float(confs_arr.max()), 4),
        "n_uncertain": n_uncertain,
        "pct_uncertain": round(pct_uncertain, 2),
    }

    # Per-class confidence
    print(f"\n  Per-class mean confidence:")
    per_class_conf = {}
    for lid, lname in id2label.items():
        mask = np.array(y_pred) == lid
        if mask.sum() > 0:
            mc = float(confs_arr[mask].mean())
            per_class_conf[lname] = round(mc, 4)
            print(f"    {lname:<8}: {mc:.4f}")
    metrics["confidence"]["per_class"] = per_class_conf

    # ── 6. Misclassification analysis ─────────────────────────────────────────
    if cfg["evaluation"]["show_misclassified"]:
        _header("6. Misclassification Analysis")
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        wrong_mask = y_true_arr != y_pred_arr
        wrong_idx  = np.where(wrong_mask)[0]

        print(f"  Total misclassified: {len(wrong_idx):,}  "
              f"({len(wrong_idx)/len(y_true)*100:.1f}%)\n")

        # Error type breakdown
        print("  Error type breakdown (True → Predicted):")
        error_pairs: dict = {}
        for i in wrong_idx:
            key = f"{id2label[y_true_arr[i]]} → {id2label[y_pred_arr[i]]}"
            error_pairs[key] = error_pairs.get(key, 0) + 1
        for pair, count in sorted(error_pairs.items(), key=lambda x: -x[1]):
            pct = count / len(wrong_idx) * 100
            print(f"    {pair:<20}: {count:>5,}  ({pct:.1f}%)")
        metrics["error_pairs"] = error_pairs

        # Sample misclassified rows
        n_show = min(cfg["evaluation"]["max_misclassified"], len(wrong_idx))
        if n_show > 0:
            print(f"\n  Sample misclassified rows (showing {n_show}):")
            sample_idx = wrong_idx[:n_show]
            for rank, i in enumerate(sample_idx, 1):
                row = test_df.iloc[i]
                print(f"\n  [{rank}] True={id2label[y_true_arr[i]]}  "
                      f"Pred={id2label[y_pred_arr[i]]}  "
                      f"Conf={confidences[i]:.3f}")
                print(f"      Q : {str(row['question'])[:120]}")
                print(f"      LA: {str(row.get('long_answer', ''))[:160]}")

    return metrics


# ── Save outputs ──────────────────────────────────────────────────────────────
def save_outputs(
    cfg: dict,
    test_df: pd.DataFrame,
    y_true: list,
    y_pred: list,
    confidences: list,
    metrics: dict,
) -> None:
    id2label   = {int(k): v for k, v in cfg["labels"].items()}
    report_dir = AI_DIR / cfg["output"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    if cfg["output"]["save_report"]:
        report_path = report_dir / "validation_metrics.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Metrics saved → {report_path}")

    if cfg["output"]["save_predictions"]:
        pred_df = test_df.copy()
        pred_df["true_label"]      = [id2label[l] for l in y_true]
        pred_df["predicted_label"] = [id2label[p] for p in y_pred]
        pred_df["confidence"]      = [round(c, 4) for c in confidences]
        pred_df["correct"]         = [t == p for t, p in zip(y_true, y_pred)]

        pred_path = report_dir / "validation_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"  Predictions saved → {pred_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg     = load_config()
    test_df = load_test_split(cfg)
    y_true, y_pred, confidences = run_inference(cfg, test_df)
    metrics = compute_metrics(cfg, test_df, y_true, y_pred, confidences)

    if cfg["output"]["save_report"] or cfg["output"]["save_predictions"]:
        _header("Saving Outputs")
        save_outputs(cfg, test_df, y_true, y_pred, confidences, metrics)

    _header("Validation Complete")
    print(f"  Accuracy     : {metrics['overall_accuracy']:.4f}")
    print(f"  F1 (macro)   : {metrics['f1_scores'].get('macro', 'N/A')}")
    print(f"  F1 (weighted): {metrics['f1_scores'].get('weighted', 'N/A')}")
