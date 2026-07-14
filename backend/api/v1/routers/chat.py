"""RAG chat endpoints: single-shot query, streaming query, and conversation history."""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.core.dependencies import (
    get_chat_service,
    get_conversation_service,
    get_current_user,
)
from backend.models.user import User
from backend.monitoring.metrics import RAG_QUERY_DURATION_SECONDS
from backend.schemas.chat import (
    ChatAnswer,
    ChatQuery,
    ConversationDetail,
    ConversationPublic,
)
from backend.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/query",
    response_model=ChatAnswer,
    summary="Ask a question and get a grounded answer",
)
async def query(
    payload: ChatQuery,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatAnswer:
    with RAG_QUERY_DURATION_SECONDS.time():
        return await chat_service.ask(
            current_user.id, payload.conversation_id, payload.question
        )


@router.post(
    "/stream", summary="Ask a question and stream the answer via Server-Sent Events"
)
async def stream(
    payload: ChatQuery,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    generator = chat_service.ask_stream(
        current_user.id, payload.conversation_id, payload.question
    )
    return StreamingResponse(generator, media_type="text/event-stream")


@router.get(
    "/conversations",
    response_model=list[ConversationPublic],
    summary="List the user's conversations",
)
async def list_conversations(
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_conversation_service),
) -> list[ConversationPublic]:
    conversations = await chat_service.list_conversations(current_user.id)
    return [ConversationPublic.model_validate(c) for c in conversations]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Fetch a conversation with its full message history",
)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_conversation_service),
) -> ConversationDetail:
    conversation = await chat_service.get_conversation(current_user.id, conversation_id)
    return ConversationDetail.model_validate(conversation)
