# Machine Learning & Experiments

This directory is dedicated to the **Model Fine-Tuning & Evaluation** requirements (8 points):

```plaintext
ml/
├── notebooks/           # Jupyter notebooks for EDA, tokenization, training, and evaluation
├── scripts/             # Python training / fine-tuning scripts
├── data/                # Dataset storage (raw and processed)
└── weights/             # Checkpoints and exported models (e.g., ONNX, safetensors)
```

## Workflow Guide:
1. **Explore & Prepare Dataset**: Run exploratory data analysis and tokenization notebooks.
2. **Fine-tune Transformer**: Train your selected model (BERT, RoBERTa, PhoBERT, BART, T5, etc.).
3. **Evaluate**: Compute evaluation metrics (Accuracy, F1, Loss curves, Confusion Matrix, etc.).
4. **Export Weights**: Export final model weights to `backend/weights/` to serve via the FastAPI application.
