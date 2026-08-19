import React, { useState, useEffect } from "react";
import { Sparkles, Activity, Clock, Cpu, Send, RefreshCw, Layers } from "lucide-react";
import { checkHealth, runPrediction } from "./services/api";
import type { HealthResponse, PredictionResponse } from "./types";

const EXAMPLE_PRESETS = [
  "This project demonstrates cutting-edge Transformer architecture for Natural Language Processing.",
  "The model achieved an outstanding 94.2% F1-score on the evaluation benchmark dataset.",
  "Unfortunately, the inference latency on the unquantized model was slightly higher than expected.",
];

export function App() {
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isHealthChecking, setIsHealthChecking] = useState(false);

  const fetchHealthStatus = async () => {
    setIsHealthChecking(true);
    try {
      const data = await checkHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setIsHealthChecking(false);
    }
  };

  useEffect(() => {
    fetchHealthStatus();
    const interval = setInterval(fetchHealthStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await runPrediction(inputText.trim());
      setResult(data);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to connect to the backend server.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header Section */}
      <header className="header">
        <div className="title-group">
          <h1>
            <Sparkles size={28} className="text-accent" />
            Transformer NLP Studio
          </h1>
          <p>Statistical Learning & NLP Web Application</p>
        </div>

        <div className="status-badge" onClick={fetchHealthStatus} style={{ cursor: "pointer" }} title="Click to refresh status">
          <span className={`status-dot ${health?.status === "healthy" ? "online" : "offline"}`} />
          <span>{health?.status === "healthy" ? "API Online" : "API Offline"}</span>
          <RefreshCw size={12} className={isHealthChecking ? "animate-spin" : ""} />
        </div>
      </header>

      {/* Main Grid */}
      <main className="main-grid">
        {/* Left Column: Input Panel */}
        <section className="card">
          <h2 className="card-title">
            <Send size={20} /> Input Text & Inference
          </h2>

          <div className="presets-container">
            <div className="presets-label">Quick Presets:</div>
            <div className="presets-list">
              {EXAMPLE_PRESETS.map((preset, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="preset-chip"
                  onClick={() => setInputText(preset)}
                >
                  Example {idx + 1}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={handlePredict} style={{ display: "flex", flexDirection: "column", flex: 1 }}>
            <textarea
              className="textarea-input"
              placeholder="Type or paste text to analyze with the Transformer model..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />

            <div className="action-row">
              <button
                type="submit"
                className="btn-primary"
                disabled={isLoading || !inputText.trim()}
              >
                {isLoading ? (
                  <>
                    <RefreshCw size={16} className="animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Run Model Inference
                  </>
                )}
              </button>
            </div>
          </form>
        </section>

        {/* Right Column: Results Panel */}
        <section className="card">
          <h2 className="card-title">
            <Layers size={20} /> Prediction & Analysis
          </h2>

          {error && (
            <div style={{ padding: "1rem", background: "rgba(244, 63, 94, 0.15)", border: "1px solid #f43f5e", borderRadius: "0.5rem", color: "#fda4af", fontSize: "0.9rem" }}>
              <strong>Error:</strong> {error}
              <div style={{ fontSize: "0.8rem", marginTop: "0.5rem", color: "#cbd5e1" }}>
                Make sure the FastAPI backend is running on <code>http://localhost:8000</code>.
              </div>
            </div>
          )}

          {!result && !error && (
            <div className="result-placeholder">
              <Activity size={36} opacity={0.4} />
              <p>Enter text on the left and click "Run Model Inference" to see the output.</p>
            </div>
          )}

          {result && !error && (
            <div className="result-card">
              <div className="prediction-banner">
                <div className="prediction-label">Task: {result.task}</div>
                <div className="prediction-value">
                  {typeof result.prediction === "object" ? JSON.stringify(result.prediction) : String(result.prediction)}
                </div>
              </div>

              {result.scores && result.scores.length > 0 && (
                <div className="scores-section">
                  <div className="presets-label">Confidence Distribution:</div>
                  {result.scores.map((item, idx) => (
                    <div key={idx} className="score-row">
                      <div className="score-meta">
                        <span>{item.label}</span>
                        <span>{(item.score * 100).toFixed(1)}%</span>
                      </div>
                      <div className="progress-bar-bg">
                        <div
                          className="progress-bar-fill"
                          style={{ width: `${Math.min(100, item.score * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <footer className="meta-footer">
                <span style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                  <Clock size={14} /> Latency: {result.execution_time_ms} ms
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                  <Cpu size={14} /> Model: {result.model_version}
                </span>
              </footer>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
