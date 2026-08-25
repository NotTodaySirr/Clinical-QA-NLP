import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW


class BioBERTTrainer:
    def __init__(
        self,
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        num_labels=3,
        device=None,
    ):
        """
        Initialise Transformer Sequence Classification model and tokenizer.

        Args:
            model_name:  HuggingFace model identifier (e.g., PubMedBERT or BioBERT)
            num_labels:  Number of output classes (0=No, 1=Yes, 2=Maybe)
            device:      torch.device to use; auto-detected if None
        """
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Initialising model on: {self.device}")
        print(f"Backbone model       : {model_name}")

        # Load model with a 3-class classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.model.to(self.device)

        # Save the tokenizer alongside the model for inference
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialise loss_fn as None; set by setup_training()
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None

    
    def setup_training(
        self,
        train_loader,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        epochs: int = 3,
        warmup_ratio: float = 0.1,
        lr_scheduler: str = "cosine",
    ) -> None:
        """
        Configure AdamW with weight decay, class-weighted loss, and LR scheduler.

        Args:
            train_loader:   PyTorch DataLoader for training data
            learning_rate:  Peak learning rate for AdamW
            weight_decay:   L2 regularization penalty
            epochs:         Total training epochs (needed for scheduler)
            warmup_ratio:   Fraction of total steps used for LR warm-up
            lr_scheduler:   LR schedule type ('cosine' or 'linear')
        """
        # Optimiser with weight decay
        # Separate parameters for weight decay (exclude bias and LayerNorm weights)
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        # Class weights (computed from label distribution, NOT from iterating the DataLoader — avoids exhausting it or redundant tokenisation)

        dataset = train_loader.dataset
        if hasattr(dataset, "dataframe") and "label_decision" in dataset.dataframe.columns:
            all_labels = dataset.dataframe["label_decision"].tolist()
        elif hasattr(dataset, "labels"):
            all_labels = dataset.labels
        else:
            all_labels = [int(dataset[i]["labels"]) for i in range(len(dataset))]

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(all_labels),
            y=all_labels,
        )
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)

        # LR scheduler with warm-up (Cosine or Linear) 
        total_steps  = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        if lr_scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        print(
            f"Training setup complete.\n"
            f"  Class weights : {class_weights.round(3)}\n"
            f"  Total steps   : {total_steps:,}  |  Warmup: {warmup_steps:,}\n"
            f"  LR / Scheduler: {learning_rate} ({lr_scheduler})\n"
            f"  Weight decay  : {weight_decay}"
        )


    def train_epoch(self, train_loader, max_grad_norm: float = 1.0) -> float:
        """
        Run one full training epoch.

        Args:
            train_loader:   DataLoader for training data
            max_grad_norm:  Gradient clipping norm (prevents exploding gradients)

        Returns:
            Average training loss for this epoch.
        """
        if self.loss_fn is None:
            raise RuntimeError("Call setup_training() before train_epoch().")

        self.model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, leave=True, desc="Training")

        for batch in loop:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = self.loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            loss.backward()

            # Gradient clipping — prevents exploding gradients in BERT layers
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()

            loop.set_postfix(loss=loss.item())

        return total_loss / len(train_loader)

    
    def evaluate(self, val_loader) -> Tuple[float, float, float]:
        """
        Run the validation loop and compute loss, accuracy, and Macro F1.

        Returns:
            (val_loss, val_accuracy, val_macro_f1) as floats.
        """
        if self.loss_fn is None:
            raise RuntimeError("Call setup_training() before evaluate().")

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits  = outputs.logits

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                _, preds = torch.max(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_loss = total_loss / len(val_loader)
        val_acc  = float(accuracy_score(all_labels, all_preds))
        val_macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))

        return val_loss, val_acc, val_macro_f1

    
    def execute_fine_tuning(
        self,
        train_loader,
        val_loader,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        lr_scheduler: str = "cosine",
        save_dir: str = "saved_model/pubmedbert_finetuned",
    ) -> None:
        """
        Full training loop — saves best weights + tokenizer based on highest Val Macro-F1.

        Args:
            train_loader:   DataLoader for training data
            val_loader:     DataLoader for validation data
            epochs:         Number of training epochs
            learning_rate:  Peak learning rate
            weight_decay:   Weight decay penalty
            lr_scheduler:   LR scheduler schedule type
            save_dir:       Directory to save the best model and tokenizer
        """
        self.setup_training(
            train_loader,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            lr_scheduler=lr_scheduler,
        )

        os.makedirs(save_dir, exist_ok=True)
        best_val_macro_f1 = -1.0
        best_val_loss = float("inf")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc, val_macro_f1 = self.evaluate(val_loader)

            print(f"  Train Loss : {train_loss:.4f}")
            print(
                f"  Val Loss   : {val_loss:.4f}  |  "
                f"Val Acc: {val_acc * 100:.2f}%  |  "
                f"Val Macro F1: {val_macro_f1 * 100:.2f}%"
            )

            # Checkpoint metric: save when Validation Macro-F1 reaches a new high
            if val_macro_f1 > best_val_macro_f1:
                print(
                    f"  Val Macro F1 improved ({best_val_macro_f1 * 100:.2f}% -> {val_macro_f1 * 100:.2f}%) "
                    f"— saving best checkpoint..."
                )
                best_val_macro_f1 = val_macro_f1
                best_val_loss = val_loss
                # Save model weights AND tokenizer so the checkpoint is
                # immediately loadable for inference
                self.model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)

        print(f"\nFine-tuning complete. Best model (Macro F1: {best_val_macro_f1 * 100:.2f}%) saved to: {save_dir}")


# Alias for backward compatibility
PubMedBERTTrainer = BioBERTTrainer