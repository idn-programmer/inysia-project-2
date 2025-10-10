from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..schemas.chat import ChatRequest, ChatResponse
from ..db.session import get_db
from ..db import models as orm


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
    content = last_user.content if last_user else ""
    reply = (
        "Thanks for your message. While I can’t give medical advice, common tips include: maintain a balanced diet, monitor carbohydrate intake, stay active, keep regular checkups, and manage blood pressure and weight. "
    )
    if content:
        reply += f'You asked: "{content}". '
    reply += "If you have concerning symptoms or a high predicted risk, please consult a healthcare professional."

    # store messages if user present
    if req.userId:
        for m in req.messages:
            if m.role == "user":
                db.add(
                    orm.ChatMessage(
                        user_id=req.userId, message=m.content
                    )
                )
        db.add(
            orm.ChatMessage(
                user_id=req.userId, message="", response=reply
            )
        )
        db.commit()
    return ChatResponse(reply=reply)


