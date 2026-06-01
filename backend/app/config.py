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
    chroma_dir: Path = Field(default=Path(".chroma"), alias="CHROMA_DIR")
    frontend_origin: str = Field(default="http://localhost:3000", alias="FRONTEND_ORIGIN")
    allow_whisper_fallback: bool = Field(default=True, alias="ALLOW_WHISPER_FALLBACK")
    ytdlp_cookies_from_browser: str | None = Field(default=None, alias="YTDLP_COOKIES_FROM_BROWSER")


@lru_cache
def get_settings() -> Settings:
    return Settings()
