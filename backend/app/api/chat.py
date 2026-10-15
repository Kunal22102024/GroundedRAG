"""`POST /api/chat`: the answer as a Server-Sent Events stream.

Events: ``status`` (stage updates), ``token`` (answer text), ``citations`` (cited sources),
``done`` (final answer, guard score, per-stage timings, token usage, flags) and ``error``.
The handler only parses the request and formats events; the work is in ``chat_service``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.deps import CurrentUser
from app.schemas import ChatRequest
from app.services import chat_service

router = APIRouter(tags=["chat"])

# `X-Accel-Buffering: no` tells nginx not to buffer this response (nginx.conf also turns it off).
_SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}


def format_sse(event: chat_service.SseEvent) -> str:
    """One event in the SSE wire format."""
    return f"event: {event.event}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"


@router.post(
    "/chat",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "Server-Sent Events stream"},
        404: {"description": "Conversation not found"},
    },
)
async def chat(body: ChatRequest, user: CurrentUser) -> StreamingResponse:
    """Answer a question from the documents the caller may read."""
    try:
        plan = await asyncio.to_thread(
            chat_service.prepare_chat,
            user,
            body.question,
            body.conversation_id,
            body.filters,
            body.source,
        )
    except chat_service.ConversationNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found") from None

    async def events() -> AsyncIterator[str]:
        async for event in chat_service.run_chat(plan):
            yield format_sse(event)

    return StreamingResponse(events(), media_type="text/event-stream", headers=_SSE_HEADERS)
