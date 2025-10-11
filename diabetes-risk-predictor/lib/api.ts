import { AuthService } from './auth';
import { PredictRequest, PredictResponse, ChatRequest, ChatResponse, PredictionOut, ApiError } from './types';

class ApiClient {
  private baseUrl = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...AuthService.getAuthHeaders(),
      ...options.headers,
    };

    const response = await fetch(url, {
      ...options,
      headers,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }

    return data;
  }

  async predict(data: PredictRequest): Promise<PredictResponse> {
    return this.request<PredictResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getHistory(limit: number = 50): Promise<PredictionOut[]> {
    // Use Next.js API route for proper SSR support
    const response = await fetch(`/api/history?limit=${limit}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }

    return data;
  }

  async chat(data: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>('/chat', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
}

export const apiClient = new ApiClient();
