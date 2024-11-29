"""SQLAlchemy models for tokenator."""

import uuid
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

from .utils import get_default_db_path

Base = declarative_base()

def get_engine(db_path: str = None):
    """Create SQLAlchemy engine with the given database path."""
    db_path = db_path or get_default_db_path()
    return create_engine(f"sqlite:///{db_path}", echo=False)

def get_session(db_path: str = None):
    """Create a thread-safe session factory."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return scoped_session(session_factory)

class TokenUsage(Base):
    """Model for tracking token usage."""
    
    __tablename__ = "token_usage"
    
    id = Column(Integer, primary_key=True)
    execution_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    
    # Create indexes
    __table_args__ = (
        Index('idx_created_at', 'created_at'),
        Index('idx_execution_id', 'execution_id'),
        Index('idx_provider', 'provider'),
        Index('idx_model', 'model'),
    )
    
    def to_dict(self):
        """Convert model instance to dictionary."""
        return {
            'id': self.id,
            'execution_id': self.execution_id,
            'provider': self.provider,
            'model': self.model,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens
        } 