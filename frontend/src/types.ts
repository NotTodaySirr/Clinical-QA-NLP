export interface ScoreItem {
  label: string;
  score: number;
}

export interface PredictionResponse {
  task: string;
  prediction: string | number | Record<string, unknown>;
  confidence?: number;
  scores?: ScoreItem[];
  execution_time_ms: number;
  model_version: string;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  model_loaded: boolean;
}
