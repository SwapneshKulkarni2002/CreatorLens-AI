from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from yt_dlp import YoutubeDL

from app.config import get_settings


YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})")
HASHTAG_RE = re.compile(r"#([\w\d_]+)")


@dataclass
class ExtractedVideo:
    video_id: str
    platform: str
    url: str
    transcript: str
    metadata: dict[str, Any]


def _compact_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _upload_date(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").date().isoformat()
    return text


def _hashtags(info: dict[str, Any]) -> list[str]:
    tags = info.get("tags") or []
    from_tags = [str(tag).lstrip("#") for tag in tags if str(tag).strip()]
    haystack = " ".join(str(info.get(k) or "") for k in ("title", "description", "fulltitle"))
    from_text = HASHTAG_RE.findall(haystack)
    return sorted(set(from_tags + from_text), key=str.lower)


def _engagement_rate(likes: int | None, comments: int | None, views: int | None) -> float | None:
    if not views:
        return None
    return round((((likes or 0) + (comments or 0)) / views) * 100, 3)


def _youtube_id(url: str) -> str | None:
    match = YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


def _base_ytdlp_opts() -> dict[str, Any]:
    settings = get_settings()
    opts: dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extract_flat": False,
    }
    if settings.ytdlp_cookies_from_browser:
        opts["cookiesfrombrowser"] = (settings.ytdlp_cookies_from_browser,)
    return opts


def _youtube_oembed_metadata(url: str, fallback_video_id: str) -> dict[str, Any]:
    response = httpx.get(
        "https://www.youtube.com/oembed",
        params={"url": url, "format": "json"},
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "id": _youtube_id(url) or fallback_video_id,
        "title": data.get("title") or "YouTube video",
        "uploader": data.get("author_name") or "Unknown creator",
        "uploader_url": data.get("author_url"),
        "webpage_url": url,
        "view_count": None,
        "like_count": None,
        "comment_count": None,
        "description": "",
        "tags": [],
        "duration": None,
        "upload_date": None,
    }


def _transcript_from_youtube_api(url: str) -> str | None:
    video_id = _youtube_id(url)
    if not video_id:
        return None
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return TextFormatter().format_transcript(transcript).strip()


def _transcript_from_subtitles(info: dict[str, Any]) -> str | None:
    subtitles = info.get("subtitles") or info.get("automatic_captions") or {}
    for language in ("en", "en-US", "en-GB"):
        tracks = subtitles.get(language) or []
        for track in tracks:
            if track.get("ext") in {"vtt", "srt", "json3"} and track.get("url"):
                response = httpx.get(track["url"], timeout=20)
                response.raise_for_status()
                return _clean_caption_text(response.text)
    return None


def _clean_caption_text(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.upper().startswith(("WEBVTT", "NOTE")):
            continue
        if "-->" in stripped or stripped.isdigit():
            continue
        stripped = re.sub(r"<[^>]+>", "", stripped)
        stripped = re.sub(r"\{.*?\}", "", stripped)
        if stripped:
            lines.append(stripped)
    return " ".join(lines).strip()


def _transcript_from_whisper(url: str) -> str | None:
    settings = get_settings()
    if not settings.allow_whisper_fallback or not settings.openai_api_key:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        output = str(Path(tmpdir) / "audio.%(ext)s")
        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "outtmpl": output,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        }
        with YoutubeDL(opts) as ydl:
            ydl.download([url])

        audio_files = list(Path(tmpdir).glob("audio.*"))
        if not audio_files:
            return None

        client = OpenAI(api_key=settings.openai_api_key)
        with audio_files[0].open("rb") as audio:
            result = client.audio.transcriptions.create(model="whisper-1", file=audio)
        return result.text.strip()


def _metadata(video_id: str, platform: str, url: str, info: dict[str, Any]) -> dict[str, Any]:
    views = _compact_int(info.get("view_count"))
    likes = _compact_int(info.get("like_count"))
    comments = _compact_int(info.get("comment_count"))
    return {
        "video_id": video_id,
        "platform": platform,
        "url": url,
        "title": info.get("title") or info.get("fulltitle"),
        "creator": info.get("uploader") or info.get("channel") or info.get("creator"),
        "creator_url": info.get("uploader_url") or info.get("channel_url"),
        "follower_count": _compact_int(info.get("channel_follower_count") or info.get("uploader_followers")),
        "views": views,
        "likes": likes,
        "comments": comments,
        "hashtags": _hashtags(info),
        "upload_date": _upload_date(info.get("upload_date") or info.get("timestamp")),
        "duration_seconds": info.get("duration"),
        "engagement_rate": _engagement_rate(likes, comments, views),
    }


def extract_video(url: str, video_id: str, platform: str) -> ExtractedVideo:
    transcript = None
    if platform == "youtube":
        try:
            transcript = _transcript_from_youtube_api(url)
        except Exception:
            transcript = None

    try:
        with YoutubeDL(_base_ytdlp_opts()) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        if platform != "youtube":
            raise exc
        try:
            info = _youtube_oembed_metadata(url, video_id)
        except Exception:
            info = {
                "id": _youtube_id(url) or video_id,
                "title": "YouTube video",
                "uploader": "Unknown creator",
                "webpage_url": url,
                "view_count": None,
                "like_count": None,
                "comment_count": None,
                "description": "",
                "tags": [],
            }

    try:
        transcript = transcript or _transcript_from_subtitles(info)
    except Exception:
        transcript = transcript

    if not transcript:
        try:
            transcript = _transcript_from_whisper(url)
        except Exception:
            transcript = None
    if not transcript:
        title = info.get("title") or "Untitled"
        description = info.get("description") or ""
        transcript = f"{title}\n\n{description}".strip()

    return ExtractedVideo(
        video_id=video_id,
        platform=platform,
        url=url,
        transcript=transcript,
        metadata=_metadata(video_id, platform, url, info),
    )
