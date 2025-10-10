from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID


class BaseResponse(BaseModel):
    message: Optional[str] = Field(default=None)


class HistoryQuery(BaseModel):
    user_id: Optional[UUID] = None
    limit: int = 50


