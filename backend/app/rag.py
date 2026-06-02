from __future__ import annotations

import json
import uuid
from collections import defaultdict
from typing import AsyncIterator

import httpx
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.config import get_settings
from app.extractors import ExtractedVideo


SESSION_MEMORY: dict[str, list[tuple[str, str]]] = defaultdict(list)
SESSION_METRICS: dict[str, list[dict]] = {}
SESSION_STORES: dict[str, QdrantVectorStore] = {}


def new_session_id() -> str:
    return uuid.uuid4().hex


class NvidiaNIMEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.url = f"{base_url.rstrip('/')}/embeddings"

    def _embed(self, text: str, input_type: str) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "input": text,
            "model": self.model,
            "input_type": input_type
        }
        response = httpx.post(self.url, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # For documents/passage indexing, use input_type="passage"
        return [self._embed(text, "passage") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        # For search querying, use input_type="query"
        return self._embed(text, "query")

def _embeddings():
    settings = get_settings()
    if settings.nvidia_api_key:
        return NvidiaNIMEmbeddings(
            model=settings.nvidia_embedding_model,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_embedding_base_url,
        )
    if settings.openai_api_key:
        return OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
    raise RuntimeError("NVIDIA_API_KEY or OPENAI_API_KEY is required to embed transcripts for this demo.")


def _collection(session_id: str) -> QdrantVectorStore:
    if session_id not in SESSION_STORES:
        raise RuntimeError("No vector index exists for this session yet. Analyze videos first.")
    return SESSION_STORES[session_id]


def _clean_metadata(metadata: dict) -> dict:
    clean = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, list):
            clean[key] = ", ".join(str(item) for item in value)
        else:
            clean[key] = str(value)
    return clean


def _doc_metadata(session_id: str, video: ExtractedVideo, chunk_index: int) -> dict:
    return _clean_metadata({
        **video.metadata,
        "session_id": session_id,
        "video_id": video.video_id,
        "chunk_index": chunk_index,
        "source": f"Video {video.video_id}, chunk {chunk_index}",
    })


def index_videos(session_id: str, videos: list[ExtractedVideo]) -> int:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=160)
    docs: list[Document] = []

    for video in videos:
        chunks = splitter.split_text(video.transcript)
        for index, chunk in enumerate(chunks, start=1):
            docs.append(Document(page_content=chunk, metadata=_doc_metadata(session_id, video, index)))

    if docs:
        SESSION_STORES[session_id] = QdrantVectorStore.from_documents(
            docs,
            embedding=_embeddings(),
            location=":memory:",
            collection_name=f"creatorlens_{session_id}",
        )
    SESSION_METRICS[session_id] = [video.metadata for video in videos]
    SESSION_MEMORY[session_id].clear()
    return len(docs)


def _format_history(session_id: str) -> str:
    history = SESSION_MEMORY.get(session_id, [])[-6:]
    return "\n".join(f"User: {user}\nAssistant: {assistant}" for user, assistant in history)


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown source")
        blocks.append(f"[{source}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _format_metrics(session_id: str) -> str:
    return json.dumps(SESSION_METRICS.get(session_id, []), indent=2)


def _fallback_answer(question: str, docs: list[Document], session_id: str) -> str:
    metrics = SESSION_METRICS.get(session_id, [])
    citations = sorted({doc.metadata.get("source", "retrieved chunk") for doc in docs})
    return (
        "LLM streaming is disabled because no NVIDIA_API_KEY or OPENAI_API_KEY is configured. "
        "Retrieved evidence is available, so here is the dynamic context to inspect.\n\n"
        f"Question: {question}\n\n"
        f"Video metrics:\n{json.dumps(metrics, indent=2)}\n\n"
        f"Citations: {', '.join(citations)}"
    )


async def stream_answer(session_id: str, question: str) -> AsyncIterator[str]:
    vectorstore = _collection(session_id)
    docs = vectorstore.similarity_search(question, k=6)

    settings = get_settings()
    citations = sorted({doc.metadata.get("source", "unknown source") for doc in docs})

    api_key = settings.nvidia_api_key or settings.openai_api_key
    if not api_key:
        answer = _fallback_answer(question, docs, session_id)
        SESSION_MEMORY[session_id].append((question, answer))
        yield answer
        return

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are CreatorLens AI, a rigorous creator analytics assistant. "
                "Use only the provided transcript chunks, metrics, and chat history. "
                "Be concise, compare Video A and Video B directly, cite sources inline as "
                "(Video A, chunk 1), and call out missing platform metrics honestly.",
            ),
            (
                "human",
                "Video metrics JSON:\n{metrics}\n\n"
                "Retrieved transcript context:\n{context}\n\n"
                "Recent chat history:\n{history}\n\n"
                "Question: {question}",
            ),
        ]
    )

    if settings.nvidia_api_key:
        llm = ChatOpenAI(
            model=settings.nvidia_model,
            temperature=0.2,
            max_tokens=1024,
            model_kwargs={"top_p": 0.7},
            streaming=True,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
        )
    else:
        llm = ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=0.2,
            streaming=True,
            api_key=settings.openai_api_key,
        )
    
    chain = prompt | llm

    full = ""
    async for chunk in chain.astream(
        {
            "metrics": _format_metrics(session_id),
            "context": _format_context(docs),
            "history": _format_history(session_id),
            "question": question,
        }
    ):
        token = chunk.content or ""
        full += token
        yield token

    if citations and "Sources:" not in full:
        source_line = "\n\nSources: " + ", ".join(citations)
        full += source_line
        yield source_line

    SESSION_MEMORY[session_id].append((question, full))
