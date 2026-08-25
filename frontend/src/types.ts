export interface ScoreItem {
  label: "Yes" | "No" | "Maybe" | string;
  score: number;
}

export interface RetrievedCandidateItem {
  rank: number;
  pubid?: number | null;
  question?: string | null;
  context: string;
  similarity_score: number;
}

export interface PredictionRequest {
  question: string;
  context?: string;
  top_k?: number;
}

export interface PredictionResponse {
  task: string;
  question: string;
  prediction: "Yes" | "No" | "Maybe" | string;
  confidence: number;
  scores: ScoreItem[];
  retrieved_context: string;
  candidates: RetrievedCandidateItem[];
  mode: "retriever_reader" | "direct_reading" | string;
  retrieval_time_ms: number;
  inference_time_ms: number;
  total_time_ms: number;
  device: string;
  model_name: string;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  model_loaded: boolean;
  retriever_loaded: boolean;
  num_indexed_contexts: number;
  device: string;
}

export interface ChatMessage {
  id: string;
  timestamp: string;
  question: string;
  customContext?: string;
  response?: PredictionResponse;
  status: "pending" | "completed" | "error";
  stepText?: string;
  errorMessage?: string;
}
