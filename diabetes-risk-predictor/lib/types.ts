// Types that match backend schemas

export interface UserSignup {
  username: string;
  email?: string;
  password: string;
}

export interface UserLogin {
  username: string;
  password: string;
}

export interface UserResponse {
  id: number;
  username: string;
  email?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: UserResponse;
}

export interface PredictRequest {
  age?: number;
  gender?: "Male" | "Female";
  pulseRate?: number;
  sbp?: number;
  dbp?: number;
  glucose?: number;
  heightCm?: number;
  weightKg?: number;
  bmi?: number;
  familyDiabetes?: boolean;
  hypertensive?: boolean;
  familyHypertension?: boolean;
  cardiovascular?: boolean;
  stroke?: boolean;
  userId?: number;
}

export interface PredictResponse {
  risk: number;
  model_version: string;
  shap_values: Record<string, number>;
  global_importance: Record<string, number>;
}

export interface PredictionOut {
  id: string;
  risk: number;
  model_version: string;
  created_at: string;
}

export interface ChatMessageIn {
  role: "user" | "assistant";
  content: string;
}

export interface PredictionContext {
  risk_score: number;
  shap_values: Record<string, number>;
  features: Record<string, any>;
}

export interface ChatRequest {
  messages: ChatMessageIn[];
  userId?: number;
  threadId?: number;
  prediction_context?: PredictionContext;
}

export interface ChatResponse {
  reply: string;
}

// Frontend specific types
export interface PredictForm {
  age: number | "";
  gender: "Male" | "Female";
  pulseRate: number | "";
  sbp: number | "";
  dbp: number | "";
  glucose: number | "";
  heightCm: number | "";
  weightKg: number | "";
  bmi: number | "";
  familyDiabetes: boolean;
  hypertensive: boolean;
  familyHypertension: boolean;
  cardiovascular: boolean;
  stroke: boolean;
}

export interface UserSession {
  user: UserResponse;
  token: string;
}

export interface ApiError {
  detail: string;
}
