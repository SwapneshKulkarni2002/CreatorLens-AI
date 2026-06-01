# CreatorLens AI

Full-stack RAG chatbot for comparing a YouTube video and an Instagram Reel.

The app takes two social video URLs, pulls transcript and metadata, computes engagement rate, chunks and embeds transcripts into Qdrant, then streams cited answers through a LangChain RAG workflow.

## Stack

- Frontend: Next.js + React + TypeScript
- Backend: FastAPI
- Orchestration: LangChain
- Embeddings: OpenAI
- Vector DB: Qdrant local mode
- LLM: OpenAI chat model by default
- Transcript/metadata: `yt-dlp`, `youtube-transcript-api`, optional OpenAI Whisper fallback

## Quick Start

### 1. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy ..\.env.example .env
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Required Environment

Create `backend/.env` from `.env.example`.

```env
OPENAI_API_KEY=your_key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
FRONTEND_ORIGIN=http://localhost:3000
```

`OPENAI_API_KEY` is required for embeddings and production-quality streaming chat in this demo. A production deployment can swap in local BGE/E5 embeddings when GPU or batch CPU workers are available.

Whisper fallback requires `ffmpeg` on the machine and `ALLOW_WHISPER_FALLBACK=true`.

If YouTube blocks metadata extraction with a bot/sign-in challenge, set `YTDLP_COOKIES_FROM_BROWSER=chrome` in `backend/.env` after signing into YouTube in Chrome. The app also falls back to transcript plus public oEmbed metadata so the demo can continue when view/like/comment counts are unavailable.

## How It Works

1. User submits one YouTube URL and one Instagram Reel URL.
2. Backend extracts metadata with `yt-dlp`.
3. YouTube transcript is fetched with `youtube-transcript-api`; both platforms can fall back to `yt-dlp` subtitles or optional Whisper transcription.
4. Engagement rate is computed as `(likes + comments) / views * 100`.
5. Transcript text is chunked with LangChain and embedded into Qdrant with metadata tags:
   - `session_id`
   - `video_id` (`A` or `B`)
   - `chunk_index`
   - creator/title/platform fields
6. Chat requests retrieve relevant chunks and pass them, video metrics, and conversation history into a streaming LangChain chain.
7. Answers include citations such as `Video A, chunk 2`.

## Scale And Cost Reasoning

This is the lowest-cost high-quality architecture for a 1000 creator/day demo workload because it avoids storing raw video, stores only transcript chunks and compact metadata, uses inexpensive embeddings once per analyzed video, and keeps generation on a small high-quality model.

Recommended production setup:

- Use `text-embedding-3-small` or BGE-small for embeddings.
- Use `gpt-4o-mini` for most creator Q&A; escalate only complex strategic questions to a larger model.
- Cache extraction results by canonical video URL.
- Persist vector indexes in a managed vector DB such as Qdrant Cloud or Pinecone serverless.
- Queue video extraction/transcription jobs with retries.
- Store normalized metadata in Postgres and chunk vectors in the vector DB.
- Add background refresh for changing metrics like likes, views, and comments.

Approximate 1000 creators/day economics:

- 2 videos per creator = 2000 videos/day.
- Embedding transcripts is cheap and one-time per URL when cached.
- Chat cost dominates; use streaming, concise prompts, and short retrieved context windows.
- Open-source embeddings plus Qdrant can reduce embedding cost further, while OpenAI embeddings usually improve retrieval quality and simplify operations.

For a production version at scale, the best alternative is FastAPI + LangGraph + Postgres + Qdrant + background workers. This repo uses LangChain + Qdrant local mode for a fast, portable demo while keeping interfaces clean enough to run Qdrant as a managed service in production.

## Submission Format

Project URL: local demo / deployed URL  
Project Description: Full-stack RAG chatbot comparing YouTube and Instagram Reel performance with cited, streaming answers.  
Loom URL: add after recording the full demo  
Github repo: add repository URL after pushing
