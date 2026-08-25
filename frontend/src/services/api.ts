import axios from "axios";
import type { HealthResponse, PredictionResponse, PredictionRequest } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30000,
});

export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>("/health");
  return response.data;
};

export const runPrediction = async (
  question: string,
  context?: string,
  top_k: number = 3
): Promise<PredictionResponse> => {
  const payload: PredictionRequest = {
    question: question.trim(),
    context: context && context.trim() ? context.trim() : undefined,
    top_k,
  };
  const response = await apiClient.post<PredictionResponse>("/predict", payload);
  return response.data;
};
