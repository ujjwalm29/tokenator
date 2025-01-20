"""SQLAlchemy models for tokenator."""

from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Index
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base

from .utils import get_default_db_path
from . import state  # Import state to access db_path

Base = declarative_base()


def get_engine(db_path: Optional[str] = None):
    """Create SQLAlchemy engine with the given database path."""
    if db_path is None:
        db_path = state.db_path or get_default_db_path()  # Use state.db_path if set
    return create_engine(f"sqlite:///{db_path}", echo=False)


def get_session():
    """Create a thread-safe session factory."""
    engine = get_engine()
    # Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return scoped_session(session_factory)


class TokenUsage(Base):
    """Model for tracking token usage."""

    __tablename__ = "token_usage"

    id = Column(Integer, primary_key=True)
    execution_id = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )

    # Core metrics (mandatory)
    total_cost = Column(Integer, nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)

    # Prompt token details (optional)
    prompt_cached_input_tokens = Column(Integer, nullable=True)
    prompt_cached_creation_tokens = Column(Integer, nullable=True)
    prompt_audio_tokens = Column(Integer, nullable=True)

    # Completion token details (optional)
    completion_audio_tokens = Column(Integer, nullable=True)
    completion_reasoning_tokens = Column(Integer, nullable=True)
    completion_accepted_prediction_tokens = Column(Integer, nullable=True)
    completion_rejected_prediction_tokens = Column(Integer, nullable=True)

    # Keep existing indexes
    __table_args__ = (
        Index("idx_created_at", "created_at"),
        Index("idx_execution_id", "execution_id"),
        Index("idx_provider", "provider"),
        Index("idx_model", "model"),
    )