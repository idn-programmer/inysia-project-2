from __future__ import annotations

from typing import List, Literal, Dict, Any, Optional
from pydantic import BaseModel
from uuid import UUID


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PredictionContext(BaseModel):
    risk_score: int
    shap_values: Dict[str, float]
    features: Dict[str, Any]  # User's input values


class ChatRequest(BaseModel):
    messages: List[ChatMessageIn]
    userId: int | None = None
    threadId: int | None = None
    prediction_context: Optional[PredictionContext] = None


class ChatResponse(BaseModel):
    reply: str


