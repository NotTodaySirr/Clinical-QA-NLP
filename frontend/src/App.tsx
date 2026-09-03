import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import {
  Stethoscope,
  Activity,
  Zap,
  Cpu,
  Search,
  FileText,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Copy,
  Check,
  RefreshCw,
  Clock,
  Database,
  ExternalLink,
  PlusCircle,
  Send,
  ArrowRight,
  X,
  BookOpen,
} from "lucide-react";
import { checkHealth, runPrediction } from "./services/api";
import type { HealthResponse, ChatMessage, RetrievedCandidateItem } from "./types";

interface ClinicalPreset {
  tag: "yes" | "no" | "maybe";
  tagLabel: string;
  title: string;
  question: string;
  customContext?: string;
}

const CLINICAL_PRESETS: ClinicalPreset[] = [
  {
    tag: "yes",
    tagLabel: "YES",
    title: "Defibrillator Implantation under Sedation",
    question: "Can we implant cardioverter defibrillator under minimal sedation?",
  },
  {
    tag: "no",
    tagLabel: "NO",
    title: "Clinical Relevance of Thyroid Stunning",
    question: "Is thyroid stunning clinically relevant?",
  },
  {
    tag: "maybe",
    tagLabel: "MAYBE",
    title: "Workplace Health & Job Stress",
    question: "Does workplace health promotion contribute to job stress reduction?",
  },
  {
    tag: "yes",
    tagLabel: "YES",
    title: "Pediatric vs Adult Disaster Victims",
    question: "Do pediatric and adult disaster victims differ?",
  },
];


const LOCAL_STORAGE_KEY = "clinical_qa_session_history_v1";

export function App() {
  // Session & Chat State
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);
  const [activeCandidateIdx, setActiveCandidateIdx] = useState<number>(0);
  const [isCitationOpen, setIsCitationOpen] = useState<boolean>(false);

  // Input & Mode State
  const [inputText, setInputText] = useState("");
  const [customContext, setCustomContext] = useState("");
  const [activeMode, setActiveMode] = useState<"odqa" | "custom">("odqa");
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  // Backend Health State
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isHealthChecking, setIsHealthChecking] = useState(false);

  // Refs
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  // Save to LocalStorage whenever messages change
  useEffect(() => {
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(messages));
    } catch (e) {
      console.error("Failed to save session history to localStorage", e);
    }
  }, [messages]);

  // Auto-scroll chat to bottom on new message
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  // Auto-resize textarea
  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  };

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
    const interval = setInterval(fetchHealthStatus, 20000);
    return () => clearInterval(interval);
  }, []);

  const handleNewSession = () => {
    setMessages([]);
    setActiveMessageId(null);
    setActiveCandidateIdx(0);
    setIsCitationOpen(false);
    localStorage.removeItem(LOCAL_STORAGE_KEY);
  };

  const handleSelectPreset = (preset: ClinicalPreset) => {
    setInputText(preset.question);
    if (preset.customContext) {
      setCustomContext(preset.customContext);
      setActiveMode("custom");
    } else {
      setActiveMode("odqa");
    }
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmitQuery();
    }
  };

  const handleSubmitQuery = async () => {
    const q = inputText.trim();
    if (!q || isLoading) return;

    const messageId = "msg_" + Date.now();
    const newMsg: ChatMessage = {
      id: messageId,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      question: q,
      customContext: activeMode === "custom" && customContext.trim() ? customContext.trim() : undefined,
      status: "pending",
      stepText: "Querying 10,500 FAISS abstracts & reasoning with PubMedBERT...",
    };

    setMessages((prev) => [...prev, newMsg]);
    setActiveMessageId(messageId);
    setActiveCandidateIdx(0);
    setInputText("");
    setIsLoading(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      const contextPayload = activeMode === "custom" && customContext.trim() ? customContext.trim() : undefined;
      const data = await runPrediction(q, contextPayload, 3);

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId ? { ...msg, status: "completed", response: data } : msg
        )
      );
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Inference connection error.";
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId
            ? { ...msg, status: "error", errorMessage: errMsg }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleOpenCitationView = (msgId: string) => {
    setActiveMessageId(msgId);
    setActiveCandidateIdx(0);
    setIsCitationOpen(true);
  };

  const handleCopyText = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Currently active message and candidate study for the Citation View
  const activeMessage = messages.find((m) => m.id === activeMessageId) || (messages.length > 0 ? messages[messages.length - 1] : null);
  const activeResponse = activeMessage?.response;
  const activeCandidates = activeResponse?.candidates || [];
  const currentCandidate: RetrievedCandidateItem | null =
    activeCandidates.length > activeCandidateIdx
      ? activeCandidates[activeCandidateIdx]
      : activeResponse
      ? {
          rank: 1,
          context: activeResponse.retrieved_context,
          similarity_score: 1.0,
        }
      : null;

  // Section parser for PubMed structured abstracts
  const parseClinicalSections = (rawText: string) => {
    const sectionKeywords = ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS", "OBJECTIVE", "PATIENTS AND METHODS"];
    const regex = new RegExp(`\\b(${sectionKeywords.join("|")}):`, "gi");
    
    if (!regex.test(rawText)) {
      return <div className="raw-abstract-text">{rawText}</div>;
    }

    const tokens = rawText.split(regex);
    const sections: { title?: string; text: string }[] = [];

    for (let i = 1; i < tokens.length; i += 2) {
      const title = tokens[i].trim().toUpperCase();
      const content = tokens[i + 1] ? tokens[i + 1].trim() : "";
      if (title && content) {
        sections.push({ title, text: content });
      }
    }

    if (sections.length === 0) {
      return <div className="raw-abstract-text">{rawText}</div>;
    }

    return (
      <div className="abstract-container">
        {sections.map((sec, idx) => (
          <div key={idx} className="clinical-section-block">
            <div className="section-tag">{sec.title}</div>
            <div className="section-content">{sec.text}</div>
          </div>
        ))}
      </div>
    );
  };

  const getVerdictClass = (pred: string) => {
    const p = pred.toLowerCase();
    if (p === "yes") return "verdict-yes";
    if (p === "no") return "verdict-no";
    return "verdict-maybe";
  };

  const getVerdictIcon = (pred: string) => {
    const p = pred.toLowerCase();
    if (p === "yes") return <CheckCircle2 size={18} />;
    if (p === "no") return <XCircle size={18} />;
    return <HelpCircle size={18} />;
  };

  return (
    <div className="app-shell">
      {/* Top Navbar */}
      <header className="top-navbar">
        <div className="nav-brand">
          <h1>
            <Stethoscope size={22} style={{ color: "var(--accent-cyan)" }} />
            Clinical QA Studio
          </h1>
          <span className="nav-tagline">Open-Domain Clinical Question Answering</span>
        </div>

        <div className="nav-controls">
          {health?.num_indexed_contexts ? (
            <div className="status-badge" title="Total PubMed abstracts indexed in FAISS">
              <Database size={13} style={{ color: "var(--accent-cyan)" }} />
              <span>{health.num_indexed_contexts.toLocaleString()} Abstracts</span>
            </div>
          ) : null}

          {health?.device ? (
            <div className="status-badge" title="Hardware accelerator">
              <Cpu size={13} style={{ color: "var(--accent-teal)" }} />
              <span>{health.device.toUpperCase()} ACCEL</span>
            </div>
          ) : null}

          <div
            className="status-badge"
            onClick={fetchHealthStatus}
            style={{ cursor: "pointer" }}
            title="Click to refresh engine status"
          >
            <span className={`status-dot ${health?.status === "healthy" ? "online" : "offline"}`} />
            <span>{health?.status === "healthy" ? "Engine Ready" : "Offline"}</span>
            <RefreshCw size={11} className={isHealthChecking ? "animate-spin" : ""} />
          </div>

          <button type="button" className="btn-new-chat" onClick={handleNewSession} title="Clear workspace">
            <PlusCircle size={14} />
            <span>New Session</span>
          </button>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className={`workspace-container ${isCitationOpen ? "citation-open" : ""}`}>
        {/* ===================================================================
            MAIN CENTRALIZED CHAT AREA
            =================================================================== */}
        <section className="chat-section">
          <div className="chat-inner-wrapper">
            {/* Scrollable Message Thread */}
            <div className="chat-scroll-area" ref={chatScrollRef}>
              {messages.length === 0 && (
                <div className="welcome-hero">
                  <div className="welcome-icon-box">
                    <Stethoscope size={28} />
                  </div>
                  <h2>Clinical Question Answering</h2>
                  <p>
                    Ask clinical research questions to retrieve supporting PubMed literature and
                    predict evidence-backed decisions (Yes / No / Maybe).
                  </p>

                  <div className="presets-container">
                    <div className="presets-label">Benchmark Clinical Queries:</div>
                    <div className="presets-grid">
                      {CLINICAL_PRESETS.map((preset, idx) => (
                        <button
                          key={idx}
                          type="button"
                          className="preset-chip"
                          onClick={() => handleSelectPreset(preset)}
                        >
                          <span className={`preset-tag tag-${preset.tag}`}>{preset.tagLabel}</span>
                          <span>{preset.title}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {messages.map((msg) => (
                <React.Fragment key={msg.id}>
                  {/* User Query Bubble */}
                  <div className="user-msg-bubble">
                    <div>{msg.question}</div>
                    <div className="user-msg-meta">{msg.timestamp}</div>
                  </div>

                  {/* AI Response Decision Card */}
                  {msg.status === "completed" && msg.response && (
                    <div
                      className={`ai-decision-card ${msg.id === activeMessageId && isCitationOpen ? "active-selected" : ""}`}
                    >
                      <div className="card-verdict-header">
                        <div className={`verdict-pill ${getVerdictClass(msg.response.prediction)}`}>
                          {getVerdictIcon(msg.response.prediction)}
                          <span>{msg.response.prediction}</span>
                        </div>
                        <span className="card-confidence">
                          {(msg.response.confidence * 100).toFixed(1)}% Confidence
                        </span>
                      </div>

                      {/* Probability Breakdown */}
                      <div className="mini-distribution">
                        {msg.response.scores.map((score, sIdx) => {
                          const fillStyle =
                            score.label.toLowerCase() === "yes"
                              ? "var(--verdict-yes)"
                              : score.label.toLowerCase() === "no"
                              ? "var(--verdict-no)"
                              : "var(--verdict-maybe)";
                          return (
                            <div key={sIdx} className="mini-bar-row">
                              <span className="mini-bar-label">{score.label}</span>
                              <div className="mini-bar-track">
                                <div
                                  className="mini-bar-fill"
                                  style={{
                                    width: `${Math.max(3, score.score * 100)}%`,
                                    backgroundColor: fillStyle,
                                  }}
                                />
                              </div>
                              <span className="mini-bar-val">{(score.score * 100).toFixed(0)}%</span>
                            </div>
                          );
                        })}
                      </div>

                      {/* View in Citation View Button */}
                      <button
                        type="button"
                        className="card-citation-btn"
                        onClick={() => handleOpenCitationView(msg.id)}
                      >
                        <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                          <BookOpen size={15} />
                          {msg.response.candidates && msg.response.candidates[0]?.pubid
                            ? `Cited Evidence (PMID: ${msg.response.candidates[0].pubid})`
                            : "Cited Evidence Document"}
                        </span>
                        <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                          {isCitationOpen && msg.id === activeMessageId ? "Viewing Citation" : "View in Reader"}
                          <ArrowRight size={14} />
                        </span>
                      </button>
                    </div>
                  )}

                  {/* Error Card */}
                  {msg.status === "error" && (
                    <div
                      style={{
                        background: "rgba(244, 63, 94, 0.12)",
                        border: "1px solid rgba(244, 63, 94, 0.35)",
                        borderRadius: "0.85rem",
                        padding: "0.95rem 1.1rem",
                        fontSize: "0.85rem",
                        color: "#fda4af",
                      }}
                    >
                      <strong>Inference Error:</strong> {msg.errorMessage}
                    </div>
                  )}
                </React.Fragment>
              ))}

              {/* Skeleton Loading Card while reasoning */}
              {isLoading && (
                <div className="skeleton-card">
                  <div className="step-indicator">
                    <RefreshCw size={15} className="animate-spin" />
                    <span>Searching 10,500 FAISS abstracts & reasoning with PubMedBERT...</span>
                  </div>
                  <div className="skeleton-box" style={{ height: "32px", width: "45%" }} />
                  <div className="skeleton-box" style={{ height: "14px", width: "90%" }} />
                  <div className="skeleton-box" style={{ height: "14px", width: "70%" }} />
                </div>
              )}
            </div>

            {/* Centralized Input Bar Footer */}
            <footer className="chat-input-footer">
              <div className="input-options-bar">
                <div className="mode-toggle-group">
                  <button
                    type="button"
                    className={`mode-toggle-btn ${activeMode === "odqa" ? "active" : ""}`}
                    onClick={() => setActiveMode("odqa")}
                  >
                    <Search size={12} style={{ display: "inline", marginRight: "3px" }} />
                    Auto Evidence (ODQA)
                  </button>
                  <button
                    type="button"
                    className={`mode-toggle-btn ${activeMode === "custom" ? "active" : ""}`}
                    onClick={() => setActiveMode("custom")}
                  >
                    <FileText size={12} style={{ display: "inline", marginRight: "3px" }} />
                    Custom Abstract
                  </button>
                </div>

                <span style={{ fontSize: "0.725rem", color: "var(--text-muted)" }}>
                  Press <strong>Enter ↵</strong> to send
                </span>
              </div>

              {/* Custom Abstract Input (if in custom mode) */}
              {activeMode === "custom" && (
                <textarea
                  className="auto-expanding-textarea"
                  style={{
                    background: "rgba(18, 26, 42, 0.8)",
                    border: "1px solid rgba(148, 163, 184, 0.15)",
                    borderRadius: "0.6rem",
                    padding: "0.6rem",
                    fontSize: "0.85rem",
                    marginBottom: "0.35rem",
                  }}
                  placeholder="Paste clinical study abstract text here..."
                  value={customContext}
                  onChange={(e) => setCustomContext(e.target.value)}
                  rows={3}
                />
              )}

              <div className="input-box-wrapper">
                <textarea
                  ref={textareaRef}
                  className="auto-expanding-textarea"
                  placeholder={
                    activeMode === "odqa"
                      ? "Ask a clinical question (e.g. Is aspirin effective for acute myocardial infarction?)..."
                      : "Enter your question for the custom abstract above..."
                  }
                  value={inputText}
                  disabled={isLoading}
                  onChange={(e) => {
                    setInputText(e.target.value);
                    adjustTextareaHeight();
                  }}
                  onKeyDown={handleKeyDown}
                  rows={1}
                />
                <button
                  type="button"
                  className="btn-send"
                  disabled={isLoading || !inputText.trim()}
                  onClick={handleSubmitQuery}
                  title="Send query (Enter)"
                >
                  <Send size={15} />
                </button>
              </div>
            </footer>
          </div>
        </section>

        {/* ===================================================================
            CITATION VIEW (Collapsible Document Reader Drawer)
            =================================================================== */}
        {isCitationOpen && activeResponse && currentCandidate && (
          <aside className="citation-sidebar">
            {/* Citation Header */}
            <div className="citation-header">
              <div className="citation-header-top">
                <div className="citation-title-group">
                  <h3>
                    {currentCandidate.question
                      ? currentCandidate.question
                      : activeResponse.question}
                  </h3>
                  <div className="citation-meta-tags">
                    {currentCandidate.pubid ? (
                      <a
                        href={`https://pubmed.ncbi.nlm.nih.gov/${currentCandidate.pubid}/`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="doc-pill doc-pill-pmid font-mono"
                        title="Open original study on PubMed NCBI"
                      >
                        <span>PMID: {currentCandidate.pubid}</span>
                        <ExternalLink size={11} />
                      </a>
                    ) : null}

                    {currentCandidate.similarity_score !== undefined && (
                      <span className="doc-pill doc-pill-match font-mono">
                        {(currentCandidate.similarity_score * 100).toFixed(1)}% Match
                      </span>
                    )}

                    <span
                      className="doc-pill"
                      style={{ background: "rgba(51, 65, 85, 0.4)", color: "var(--text-secondary)" }}
                    >
                      Rank #{currentCandidate.rank || 1}
                    </span>
                  </div>
                </div>

                <div className="citation-actions">
                  <button
                    type="button"
                    className="btn-icon-action"
                    onClick={() => handleCopyText(currentCandidate.context)}
                    title="Copy abstract text"
                  >
                    {copied ? <Check size={13} /> : <Copy size={13} />}
                    <span>{copied ? "Copied" : "Copy"}</span>
                  </button>

                  <button
                    type="button"
                    className="btn-close-citation"
                    onClick={() => setIsCitationOpen(false)}
                    title="Close citation view"
                  >
                    <X size={16} />
                  </button>
                </div>
              </div>

              {/* Candidate Studies Tabs (if multiple candidates) */}
              {activeCandidates.length > 1 && (
                <div className="candidate-tabs-bar">
                  {activeCandidates.map((cand, idx) => (
                    <button
                      key={idx}
                      type="button"
                      className={`cand-tab ${idx === activeCandidateIdx ? "active" : ""}`}
                      onClick={() => setActiveCandidateIdx(idx)}
                    >
                      Candidate #{cand.rank} {cand.pubid ? `(PMID ${cand.pubid})` : ""}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Citation Body with Formatted Sections */}
            <div className="citation-body">
              {parseClinicalSections(currentCandidate.context)}
            </div>

            {/* Diagnostics Footer */}
            <footer className="citation-footer">
              <div className="diag-stat-group">
                <div className="diag-stat">
                  <Clock size={13} />
                  <span>Retrieval: <strong>{activeResponse.retrieval_time_ms} ms</strong></span>
                </div>
                <div className="diag-stat">
                  <Activity size={13} />
                  <span>Reader: <strong>{activeResponse.inference_time_ms} ms</strong></span>
                </div>
                <div className="diag-stat">
                  <Zap size={13} />
                  <span>Total: <strong>{activeResponse.total_time_ms} ms</strong></span>
                </div>
              </div>

              <div className="diag-stat">
                <Cpu size={13} />
                <span>Device: <strong>{activeResponse.device.toUpperCase()}</strong></span>
              </div>
            </footer>
          </aside>
        )}
      </div>
    </div>
  );
}

export default App;
