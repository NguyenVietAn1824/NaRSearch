from __future__ import annotations

import datetime
from uuid import UUID
from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TEXT
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, relationship
from sqlmodel import Field, Relationship, SQLModel
from enum import Enum
from typing import List, Optional
from datetime import datetime as dt, timezone


class MessageRole(str, Enum):
    USER = "user"
    TOOL = "tool"
    SYSTEM = "system"


class Conversation(SQLModel, table=True):
    __tablename__ = "conversation"
    id: UUID = Field(
        sa_column=Column(
            PGUUID(as_uuid=True),
            primary_key=True,
            server_default=text("gen_random_uuid()"),
        ),
    )
    messages: Mapped[list["Message"]] = Relationship(
        sa_relationship=relationship(back_populates="conversation")
    )

    message_runs: Mapped[List["MessageRun"]] = Relationship(
        sa_relationship=relationship(back_populates="conversation")
    )
    
    created_at: dt = Field(default_factory=lambda: dt.now(timezone.utc))


class MessageRun(SQLModel, table=True):
    __tablename__ = "message_run"
    id: UUID = Field(
        sa_column=Column(
            PGUUID(as_uuid=True),
            primary_key=True,
            server_default=text("gen_random_uuid()"),
        ),
    )
    conversation_id: UUID = Field(
        foreign_key="conversation.id",
        nullable=False,
        index=True,
    )

    messages: dict | list = Field(
        sa_column=Column("messages", JSONB, nullable=False),
        default_factory=list,
    )

    conversation: Mapped["Conversation"] = Relationship(
        sa_relationship=relationship(back_populates="message_runs")
    )
    created_at: dt = Field(default_factory=lambda: dt.now(timezone.utc))

class Message(SQLModel, table=True):
    __tablename__ = "message"
    id: UUID = Field(
        sa_column=Column(
            PGUUID(as_uuid=True),
            primary_key=True,
            server_default=text("gen_random_uuid()"),
        ),
    )
    conversation_id: UUID = Field(
        foreign_key="conversation.id",
        nullable=False,
        index=True,
    )
    role: MessageRole
    content: str
    metadata_: Optional[dict] = Field(
        default=None,
        sa_column=Column("metadata", JSONB),
    )
    created_at: dt = Field(default_factory=lambda: dt.now(timezone.utc))

    conversation: Mapped["Conversation"] = Relationship(
        sa_relationship=relationship(back_populates="messages")
    )