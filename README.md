# Clinical Question Answering System

A full-stack web application for **Biomedical & Clinical Question Answering (QA)**, built with **FastAPI** on the backend and **React (Vite + TypeScript)** on the frontend.

The project uses biomedical literature datasets from **PubMed** (PubMedQA) and features a fine-tuned **PubMedBERT** model trained on a **10.5K labeled dataset** (combining pseudo-labeled and gold-standard clinical QA pairs) to deliver accurate clinical reasoning (`Yes` / `No` / `Maybe`), supported by vector-based context retrieval (FAISS) and probability confidence scoring.


---

## 📁 Project Structure

```plaintext
.
├── backend/                  # FastAPI Application
│   ├── app/
│   │   ├── api/routes.py     # Endpoints (/health, /predict)
│   │   ├── core/config.py    # Configuration & CORS settings
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   ├── services/         # Pluggable model inference engine
│   │   └── main.py           # FastAPI entrypoint & lifespan
│   ├── weights/              # Exported model weights (.safetensors, .onnx)
│   ├── pyproject.toml        # uv configuration & dependencies
│   └── requirements.txt      # pip dependency list
├── frontend/                 # React 18 + TypeScript + Vite SPA
│   ├── src/
│   │   ├── services/api.ts   # Axios client for backend API
│   │   ├── types.ts          # TypeScript interfaces
│   │   ├── App.tsx           # NLP Studio UI & inference playground
│   │   └── index.css         # Modern styling & themes
│   └── package.json
├── ml/                       # Model training & fine-tuning experiments
│   ├── notebooks/            # Jupyter notebooks for EDA, training & evaluation
│   ├── data/                 # Datasets
│   └── weights/              # Checkpoint artifacts
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Backend Setup

You can run the backend using either **`uv`** (recommended) or traditional **`pip`**:

#### Option A: Using `uv` (Recommended - Fastest)

```bash
# Navigate to backend directory
cd backend

# Sync dependencies (creates venv and installs everything automatically)
uv sync

# Run the backend server
uv run main.py
```

> **Note**: You can also run with live-reload using `uv run uvicorn app.main:app --reload --port 8000`.


#### Option B: Using standard `pip`

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
# Windows:
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS:
# python3 -m venv .venv
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the development server
uvicorn app.main:app --reload --port 8000
```

The API will be live at:
- **API Base**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/v1/health`

---

### 2. Frontend Setup

You can run the frontend using either **`pnpm`** (recommended) or standard **`npm`**:

#### Option A: Using `pnpm` (Recommended - Fast & Disk-Efficient)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
pnpm install

# Run the development server
pnpm dev
```

#### Option B: Using standard `npm`

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

The web application will be accessible at `http://localhost:5173`.

---

### 3. Dataset Setup

#### Option A: Pull via DVC & DagsHub (Recommended)
```bash
# Install root dependencies (includes DVC and ML tools)
# Using uv:
uv venv && uv pip install -r requirements.txt
# Or using pip:
# python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt

# Configure DagsHub credentials (one-time setup)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "YOUR_DAGSHUB_USERNAME"
dvc remote modify origin --local password "YOUR_DAGSHUB_TOKEN"

# Pull data tracked by DVC
dvc pull -r origin
```

#### Option B: Fetch raw dataset directly via Python script
```bash
# Using uv:
uv run python backend/AI/src/data_setup.py

# Or using standard python:
# python backend/AI/src/data_setup.py
```


---

## 📡 API Reference

### Health Check
- **Endpoint**: `GET /api/v1/health`
- **Response**:
```json
{
  "status": "healthy",
  "app_name": "Clinical Question Answering System",
  "version": "1.0.0",
  "model_loaded": true,
  "retriever_loaded": true,
  "num_indexed_contexts": 10500,
  "device": "cuda"
}
```

### Run Model Inference
- **Endpoint**: `POST /api/v1/predict`
- **Request Body**:
```json
{
  "question": "Does hydroxychloroquine improve survival in hospitalized COVID-19 patients?",
  "context": null,
  "top_k": 3
}
```
- **Response**:
```json
{
  "task": "Clinical Question Answering",
  "question": "Does hydroxychloroquine improve survival in hospitalized COVID-19 patients?",
  "prediction": "No",
  "confidence": 0.94,
  "scores": [
    { "label": "Yes", "score": 0.03 },
    { "label": "No", "score": 0.94 },
    { "label": "Maybe", "score": 0.03 }
  ],
  "retrieved_context": "Hydroxychloroquine did not result in a significantly lower incidence of death...",
  "candidates": [],
  "mode": "retriever_reader",
  "retrieval_time_ms": 12.4,
  "inference_time_ms": 45.2,
  "total_time_ms": 57.6,
  "device": "cuda",
  "model_name": "PubMedBERT Fine-Tuned (3-class)"
}
```

---

## 🧠 Machine Learning Workflow (`ml/`)

1. **Fine-Tuning**: Place training notebooks in `ml/notebooks/`.
2. **Evaluation**: Compute required metrics (Accuracy, F1, Loss, etc.) for the project report.
3. **Model Integration**: Export the fine-tuned weights/pipeline to `backend/weights/` and update `backend/app/services/inference.py` to load your trained model.
