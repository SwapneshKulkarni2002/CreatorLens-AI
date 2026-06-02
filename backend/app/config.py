from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    nvidia_api_key: str | None = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL")
    nvidia_model: str = Field(default="google/gemma-2-2b-it", alias="NVIDIA_MODEL")
    nvidia_embedding_model: str = Field(default="nvidia/nv-embedqa-e5-v5", alias="NVIDIA_EMBEDDING_MODEL")
    nvidia_embedding_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_EMBEDDING_BASE_URL")
    chroma_dir: Path = Field(default=Path(".chroma"), alias="CHROMA_DIR")
    frontend_origin: str = Field(default="http://localhost:3000", alias="FRONTEND_ORIGIN")
    allow_whisper_fallback: bool = Field(default=True, alias="ALLOW_WHISPER_FALLBACK")
    ytdlp_cookies_from_browser: str | None = Field(default=None, alias="YTDLP_COOKIES_FROM_BROWSER")


@lru_cache
def get_settings() -> Settings:
    return Settings()
