"""
Async SQLAlchemy engine/session setup, plus small helper functions for the
three things turn_manager.py needs: "make sure this conversation exists",
"record a turn", and "fetch recent turns for reconnect replay."
"""

import os

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from database.models import Base, Conversation, Turn

DATABASE_URL = os.environ["DATABASE_URL"]

_engine = create_async_engine(DATABASE_URL, echo=False)
_SessionLocal = async_sessionmaker(_engine, expire_on_commit=False)


async def init_db():
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_or_create_conversation(conversation_id: str) -> None:
    async with _SessionLocal() as session:
        existing = await session.get(Conversation, conversation_id)
        if existing is None:
            session.add(Conversation(id=conversation_id))
            await session.commit()


async def record_turn(
    conversation_id: str,
    input_type: str,
    user_text: str | None,
    agent_text: str | None,
    is_fallback: bool = False,
) -> None:
    async with _SessionLocal() as session:
        session.add(
            Turn(
                conversation_id=conversation_id,
                input_type=input_type,
                user_text=user_text,
                agent_text=agent_text,
                is_fallback=is_fallback,
            )
        )
        await session.commit()


async def get_recent_turns(conversation_id: str, limit: int = 6) -> list[dict]:
    
    async with _SessionLocal() as session:
        result = await session.execute(
            select(Turn)
            .where(Turn.conversation_id == conversation_id)
            .where(Turn.is_fallback == False)  
            .order_by(Turn.created_at.desc())
            .limit(limit)
        )
        turns = list(result.scalars().all())
        turns.reverse()  # oldest first
        return [
            {"user_text": t.user_text, "agent_text": t.agent_text} for t in turns
        ]


async def close_conversation(conversation_id: str) -> None:
    from datetime import datetime, timezone

    async with _SessionLocal() as session:
        convo = await session.get(Conversation, conversation_id)
        if convo is not None:
            convo.ended_at = datetime.now(timezone.utc)
            await session.commit()