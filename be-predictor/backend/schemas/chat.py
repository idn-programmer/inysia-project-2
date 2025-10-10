from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel
from uuid import UUID


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessageIn]
    userId: int | None = None
    threadId: int | None = None


class ChatResponse(BaseModel):
    reply: str


