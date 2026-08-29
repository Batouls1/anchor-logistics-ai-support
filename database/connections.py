"""
Async SQLAlchemy engine/session setup, plus helpers: create a
conversation, record a turn, mark a conversation closed.
"""

import os
from datetime import datetime, timezone

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


async def load_recent_turns(conversation_id: str, limit: int = 20) -> list[tuple[str, str]]:
    """
    Returns the last `limit` completed exchanges as (user_text, agent_text),
    oldest first, so an in-memory session can be rebuilt from durable
    storage on any process.

    Fallback turns are skipped -- "I didn't catch that" was never a real
    exchange and replaying it would teach the model a conversation that
    didn't happen.
    """
    async with _SessionLocal() as session:
        result = await session.execute(
            select(Turn.user_text, Turn.agent_text)
            .where(
                Turn.conversation_id == conversation_id,
                Turn.is_fallback.is_(False),
                Turn.user_text.isnot(None),
                Turn.agent_text.isnot(None),
            )
            .order_by(Turn.created_at.desc())
            .limit(limit)
        )
        rows = result.all()

    # Fetched newest-first so the limit keeps the *most recent* turns;
    # reversed here because history has to replay in chronological order.
    return [(user_text, agent_text) for user_text, agent_text in reversed(rows)]


async def close_conversation(conversation_id: str) -> None:
    async with _SessionLocal() as session:
        convo = await session.get(Conversation, conversation_id)
        if convo is not None:
            convo.ended_at = datetime.now(timezone.utc)
            await session.commit()