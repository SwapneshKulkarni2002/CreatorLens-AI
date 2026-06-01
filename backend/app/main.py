from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.extractors import extract_video
from app.models import AnalyzeRequest, AnalyzeResponse, ChatRequest, VideoMetrics
from app.rag import index_videos, new_session_id, stream_answer


settings = get_settings()
app = FastAPI(title="CreatorLens AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    session_id = request.session_id or new_session_id()
    try:
        videos = [
            extract_video(str(request.youtube_url), "A", "youtube"),
            extract_video(str(request.instagram_url), "B", "instagram"),
        ]
        chunk_count = index_videos(session_id, videos)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not analyze videos: {exc}") from exc

    return AnalyzeResponse(
        session_id=session_id,
        videos=[
            VideoMetrics(
                **video.metadata,
                transcript_preview=video.transcript[:320],
                transcript_char_count=len(video.transcript),
            )
            for video in videos
        ],
        chunk_count=chunk_count,
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def events():
        async for token in stream_answer(request.session_id, request.message):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")

