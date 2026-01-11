from __future__ import annotations

from litellm import Iterable

from backend.cores.pg.model import Conversation, Message, MessageRun
from .base_database import BaseRepository
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import UUID, select


class ConversationRepository(BaseRepository[Conversation]):

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)
        
    async def get_conversation(self, conversation_id: UUID) -> Conversation | None:
        return await self.session.get(Conversation, conversation_id)

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        self.add(conversation)
        self.flush()
        return conversation

    async def delete_conversation(self, conversation_id: UUID) -> None:
        conversation = await self.session.get(Conversation, conversation_id)
        if conversation:
            await self.session.delete(conversation)
            await self.session.commit()

    async def add_messages(
    self, conversation: Conversation, messages: Iterable[Message]
    ) -> list[Message]:
        for m in messages:
            m.conversation_id = conversation.id
            await self.add(m)
        await self.flush()
        return list(messages)
    
    async def add_message_run(
        self, conversation: Conversation, messages_obj: dict | list
    ) -> MessageRun:
        run = MessageRun(
            conversation_id=conversation.id,
            messages=messages_obj,
        )
        await self.add(run)
        await self.flush()
        await self.refresh(run)
        return run