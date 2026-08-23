"""
run.py
======
Fine-tuning entry point for the Clinical QA BioBERT model.

All hyperparameters are loaded from:
  backend/AI/config/training_config.yaml

Run order
---------
  1. python backend/AI/src/data_process/main.py   <- data prep + FAISS index
  2. python backend/AI/src/fine_tuning/run.py      <- this file (training only)

Prerequisites
-------------
  pubmedqa_pseudo_labeled.csv must exist in data/labeled/
  (run backend/AI/src/label_pipeline/llm_labeler.py first)
"""

import os
import sys
from pathlib import Path

import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
# run.py lives at: <project>/backend/AI/src/fine_tuning/run.py
PROJECT_ROOT = Path(__file__).resolve().parents[4]   # …/Clinical-QA-NLP
AI_DIR       = PROJECT_ROOT / "backend" / "AI"
SRC_DIR      = AI_DIR / "src"
CONFIG_PATH  = AI_DIR / "config" / "training_config.yaml"

# Add AI/src so that package-relative imports inside each module resolve:
#   data_process/data_processor.py  uses  from .data_ingestion import ...
#   fine_tuning/trainer.py          is    imported as fine_tuning.trainer
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_process.data_processor import DataProcessor   # noqa: E402
from fine_tuning.trainer import BioBERTTrainer           # noqa: E402


def load_config(config_path: Path) -> dict:
    """Load and return the YAML training config."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found:\n  {config_path}\n"
            "Expected: backend/AI/config/training_config.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"Config loaded from: {config_path}\n")
    return cfg


# ── Step 1: Load & Process Labeled Data ──────────────────────────────────────
def load_and_process(cfg: dict) -> tuple:
    """
    Load pseudo-labeled CSV, preprocess labels, stratified-split,
    and return (train_loader, val_loader, test_loader).
    """
    print("=" * 60)
    print("Loading and processing labeled training data")
    print("=" * 60)

    labeled_csv = AI_DIR / cfg["data"]["labeled_csv"]
    if not labeled_csv.exists():
        raise FileNotFoundError(
            f"Labeled CSV not found:\n  {labeled_csv}\n\n"
            "Run the following first:\n"
            "  python backend/AI/src/label_pipeline/llm_labeler.py"
        )

    processor = DataProcessor(
        cleaned_csv_path=str(labeled_csv),
        model_name=cfg["model"]["name"],
        max_len=cfg["model"]["max_len"],
    )
    processor.preprocess()
    processor.stratified_split(
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )
    train_loader, val_loader, test_loader = processor.get_dataloader(
        batch_size=cfg["training"]["batch_size"]
    )

    return train_loader, val_loader, test_loader


# ── Step 2: Fine-tune BioBERT ─────────────────────────────────────────────────
def fine_tune(cfg: dict, train_loader, val_loader) -> None:
    """
    Initialise BioBERTTrainer and run the full fine-tuning loop.
    Best model checkpoint + tokenizer are saved to output.save_dir.
    """
    print("\n" + "=" * 60)
    print("Fine-tuning BioBERT")
    print("=" * 60)

    save_dir = str(AI_DIR / cfg["output"]["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    trainer = BioBERTTrainer(
        model_name=cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
    )
    trainer.execute_fine_tuning(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        save_dir=save_dir,
    )

    return save_dir


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)

    train_loader, val_loader, test_loader = load_and_process(cfg)
    save_dir = fine_tune(cfg, train_loader, val_loader)

    print("\n" + "=" * 60)
    print("Fine-tuning complete.")
    print(f"  Config used      : {CONFIG_PATH}")
    print(f"  Model checkpoint : {save_dir}")
    print("=" * 60)
