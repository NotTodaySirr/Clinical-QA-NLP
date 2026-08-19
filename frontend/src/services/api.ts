import axios from "axios";
import type { HealthResponse, PredictionResponse } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 10000,
});

export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>("/health");
  return response.data;
};

export const runPrediction = async (text: string, parameters: Record<string, unknown> = {}): Promise<PredictionResponse> => {
  const response = await apiClient.post<PredictionResponse>("/predict", {
    text,
    parameters,
  });
  return response.data;
};
