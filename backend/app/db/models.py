from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ClarificationSession(Base):
    __tablename__ = "clarification_sessions"

    id = Column(String, primary_key=True)
    requirement_doc = Column(Text, nullable=False)
    requirement_doc_summary = Column(Text)
    status = Column(String, default="running")
    created_at = Column(DateTime, server_default="now()")
    updated_at = Column(DateTime, server_default="now()", onupdate="now()")

    rounds = relationship("ClarificationRound", back_populates="session")


class ClarificationRound(Base):
    __tablename__ = "clarification_rounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("clarification_sessions.id"))
    round_number = Column(Integer)
    questions = Column(JSON)
    filtered_questions = Column(JSON)
    passed = Column(Boolean, default=False)

    session = relationship("ClarificationSession", back_populates="rounds")
