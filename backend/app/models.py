from pydantic import BaseModel, HttpUrl


class AnalyzeRequest(BaseModel):
    youtube_url: HttpUrl
    instagram_url: HttpUrl
    session_id: str | None = None


class VideoMetrics(BaseModel):
    video_id: str
    platform: str
    url: str
    title: str | None = None
    creator: str | None = None
    creator_url: str | None = None
    follower_count: int | None = None
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    hashtags: list[str] = []
    upload_date: str | None = None
    duration_seconds: float | None = None
    engagement_rate: float | None = None
    transcript_preview: str = ""
    transcript_char_count: int = 0
    transcript: str = ""


class AnalyzeResponse(BaseModel):
    session_id: str
    videos: list[VideoMetrics]
    chunk_count: int


class ChatRequest(BaseModel):
    session_id: str
    message: str

