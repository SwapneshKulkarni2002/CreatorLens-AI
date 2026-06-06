"use client";

import { FormEvent, useMemo, useState } from "react";
import { BarChart3, Clock, Loader2, MessageSquare, Play, Send, Sparkles, TrendingUp } from "lucide-react";

type VideoMetrics = {
  video_id: string;
  platform: string;
  url: string;
  title?: string | null;
  creator?: string | null;
  creator_url?: string | null;
  follower_count?: number | null;
  views?: number | null;
  likes?: number | null;
  comments?: number | null;
  thumbnail?: string | null;
  hashtags: string[];
  upload_date?: string | null;
  duration_seconds?: number | null;
  engagement_rate?: number | null;
  transcript_preview: string;
  transcript_char_count: number;
  transcript: string;
};

type AnalyzeResponse = {
  session_id: string;
  videos: VideoMetrics[];
  chunk_count: number;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

function formatNumber(value?: number | null) {
  if (value === null || value === undefined) return "Unknown";
  return new Intl.NumberFormat("en", { notation: value > 9999 ? "compact" : "standard" }).format(value);
}

function formatRate(value?: number | null) {
  if (value === null || value === undefined) return "Unknown";
  return `${value.toFixed(2)}%`;
}

function getInstagramEmbedUrl(urlStr: string): string {
  try {
    const url = new URL(urlStr);
    const match = url.pathname.match(/\/(reel|p|tv)\/([A-Za-z0-9_-]+)/);
    if (match) {
      const type = match[1];
      const shortcode = match[2];
      return `https://www.instagram.com/${type}/${shortcode}/embed/`;
    }
    let pathname = url.pathname;
    if (!pathname.endsWith('/')) {
      pathname += '/';
    }
    return `${url.origin}${pathname}embed/`;
  } catch (e) {
    return urlStr;
  }
}

function getYouTubeEmbedUrl(urlStr: string): string {
  try {
    const url = new URL(urlStr);
    if (url.hostname.includes("youtu.be")) {
      const videoId = url.pathname.slice(1);
      return `https://www.youtube.com/embed/${videoId}`;
    }
    if (url.pathname.startsWith("/shorts/")) {
      const videoId = url.pathname.split("/")[2];
      return `https://www.youtube.com/embed/${videoId}`;
    }
    const videoId = url.searchParams.get("v");
    if (videoId) {
      return `https://www.youtube.com/embed/${videoId}`;
    }
    return urlStr;
  } catch (e) {
    return urlStr;
  }
}

function VideoCard({ video }: { video: VideoMetrics }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const showReadMore = video.transcript && video.transcript.length > video.transcript_preview.length;

  const isInstagram = video.platform.toLowerCase() === "instagram";
  const isYouTube = video.platform.toLowerCase() === "youtube";

  return (
    <article className="video-card">
      <div className="card-shine" />
      {isInstagram ? (
        <div className="video-card__thumbnail embed">
          <iframe
            src={getInstagramEmbedUrl(video.url)}
            className="embed-iframe"
            title="Instagram Reel Preview"
            scrolling="no"
            allowFullScreen
            allow="autoplay; clipboard-write; encrypted-media; picture-in-picture; web-share"
          />
        </div>
      ) : isYouTube ? (
        <div className="video-card__thumbnail embed">
          <iframe
            src={getYouTubeEmbedUrl(video.url)}
            className="embed-iframe"
            title="YouTube Video Preview"
            allowFullScreen
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          />
        </div>
      ) : video.thumbnail ? (
        <div className="video-card__thumbnail">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <a href={video.url} target="_blank" rel="noreferrer">
            <img className="thumbnail-img" src={video.thumbnail} alt={video.title || "Video thumbnail"} loading="lazy" referrerPolicy="no-referrer" />
          </a>
        </div>
      ) : (
        <div className="video-card__thumbnail placeholder">
          <span>No thumbnail</span>
        </div>
      )}
      <div className="video-card__top">
        <div>
          <span className={`platform-badge ${video.platform.toLowerCase()}`}>
            {video.platform}
          </span>
          <p className="eyebrow" style={{ marginTop: "8px" }}>Video {video.video_id}</p>
          <h2>{video.title || "Untitled video"}</h2>
        </div>
        <a href={video.url} target="_blank" rel="noreferrer" className="icon-link" aria-label={`Open Video ${video.video_id}`}>
          <Play size={16} fill="currentColor" />
        </a>
      </div>

      <div className="creator-line">
        <span className="creator-name">{video.creator || "Unknown creator"}</span>
        <span>{formatNumber(video.follower_count)} followers</span>
      </div>

      <div className="metric-grid">
        <span>Views <strong>{formatNumber(video.views)}</strong></span>
        <span>Likes <strong>{formatNumber(video.likes)}</strong></span>
        <span>Comments <strong>{formatNumber(video.comments)}</strong></span>
        <span className="metric-grid__hot">Engagement <strong>{formatRate(video.engagement_rate)}</strong></span>
      </div>

      <div className="meta-row">
        <span>{video.upload_date || "Date unknown"}</span>
        <span>{video.duration_seconds ? `${Math.round(video.duration_seconds)}s` : "Duration unknown"}</span>
        <span>{formatNumber(video.transcript_char_count)} chars</span>
      </div>

      <div className="transcript-box">
        <p className={`transcript-text ${isExpanded ? "expanded" : ""}`}>
          {isExpanded ? video.transcript : (video.transcript_preview || "No transcript preview available.")}
        </p>
        {showReadMore && (
          <button 
            type="button" 
            onClick={() => setIsExpanded(!isExpanded)} 
            className="read-more-btn"
          >
            {isExpanded ? "Show Less" : "Read Full Transcript"}
          </button>
        )}
      </div>

      <div className="tags">
        {video.hashtags.slice(0, 6).map((tag) => (
          <span key={tag}>#{tag}</span>
        ))}
      </div>
    </article>
  );
}

export default function Home() {
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [instagramUrl, setInstagramUrl] = useState("");
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState<ChatMessage[]>([]);

  const canChat = Boolean(analysis?.session_id) && !isStreaming;
  const exampleQuestions = useMemo(
    () => [
      "Why did Video A get more engagement than Video B?",
      "What's the engagement rate of each?",
      "Compare the hooks in the first 5 seconds.",
      "Who's the creator of Video B and what's their follower count?",
      "Suggest improvements for B based on what worked in A."
    ],
    []
  );

  async function analyze(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsAnalyzing(true);
    setChat([]);

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ youtube_url: youtubeUrl, instagram_url: instagramUrl })
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Video analysis failed.");
      }
      setAnalysis(await response.json());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function sendChat(text = message) {
    const trimmed = text.trim();
    if (!analysis?.session_id || !trimmed || isStreaming) return;

    setMessage("");
    setError("");
    setIsStreaming(true);
    setChat((current) => [...current, { role: "user", content: trimmed }, { role: "assistant", content: "" }]);

    try {
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: analysis.session_id, message: trimmed })
      });
      if (!response.ok || !response.body) throw new Error("Chat stream failed.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";

        for (const frame of frames) {
          const line = frame.split("\n").find((entry) => entry.startsWith("data: "));
          if (!line) continue;
          const data = line.slice(6);
          if (data === "[DONE]") continue;
          const { token } = JSON.parse(data);
          setChat((current) => {
            const next = [...current];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, content: last.content + token };
            return next;
          });
        }
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Chat failed.");
    } finally {
      setIsStreaming(false);
    }
  }

  return (
    <main className="shell">
      <section className="topbar">
        <div className="topbar-left">
          <div className="topbar-logo">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img 
              src="/banner.png" 
              alt="CreatorLens AI Logo" 
              style={{ width: "70px", height: "70px", objectFit: "cover" }} 
            />
          </div>
          <div>
            <p className="eyebrow">CreatorLens AI · NVIDIA Gemma 3 NIM</p>
            <h1>Compare two creator videos with cited RAG answers.</h1>
            <p className="hero-copy">Drop in one YouTube link and one Instagram Reel. CreatorLens builds a transcript index, measures engagement, and streams source-backed answers using Gemma 3.</p>
          </div>
        </div>
        <div className="status-pill">
          <Sparkles size={16} />
          <span>Gemma 3 · Qdrant · RAG</span>
        </div>
      </section>

      <form className="input-band" onSubmit={analyze}>
        <label>
          YouTube URL
          <input value={youtubeUrl} onChange={(event) => setYoutubeUrl(event.target.value)} placeholder="https://www.youtube.com/watch?v=..." required />
        </label>
        <label>
          Instagram Reel URL
          <input value={instagramUrl} onChange={(event) => setInstagramUrl(event.target.value)} placeholder="https://www.instagram.com/reel/..." required />
        </label>
        <button type="submit" disabled={isAnalyzing}>
          {isAnalyzing ? <Loader2 className="spin" size={18} /> : <BarChart3 size={18} />}
          <span>Analyze Performance</span>
        </button>
      </form>

      <section className="proof-strip" aria-label="Workflow">
        <span><Clock size={16} /> Extract metadata & fallback gracefully</span>
        <span><TrendingUp size={16} /> Compute engagement rates</span>
        <span><MessageSquare size={16} /> Query Gemma-3 with citations</span>
      </section>

      {error && <div className="error">{error}</div>}

      <section className="workspace">
        <div className="videos">
          {analysis ? (
            <>
              <div className="analysis-meta">
                <Sparkles size={14} className="spin" />
                <span>{analysis.chunk_count} transcript chunks indexed for session {analysis.session_id.slice(0, 8)}</span>
              </div>
              <div className="video-grid">
                {analysis.videos.map((video) => <VideoCard key={video.video_id} video={video} />)}
              </div>
            </>
          ) : (
            <div className="empty-state">
              <BarChart3 size={40} />
              <h3>Comparison Index Ready</h3>
              <p>Submit one YouTube video and one Instagram Reel to build the dynamic comparison index.</p>
            </div>
          )}
        </div>

        <aside className="chat-panel">
          <div className="chat-header">
            <div>
              <p className="eyebrow">Strategic Assistant</p>
              <h2>Ask performance questions</h2>
            </div>
            <MessageSquare size={22} />
          </div>

          <div className="quick-prompts">
            {exampleQuestions.map((question) => (
              <button key={question} type="button" onClick={() => sendChat(question)} disabled={!canChat}>
                {question}
              </button>
            ))}
          </div>

          <div className="messages">
            {chat.length === 0 ? (
              <p className="hint">Answers stream with citations once analysis finishes.</p>
            ) : (
              chat.map((entry, index) => (
                <div key={`${entry.role}-${index}`} className={`message ${entry.role}`}>
                  {entry.content || (entry.role === "assistant" ? "Thinking..." : "")}
                </div>
              ))
            )}
          </div>

          <form className="chat-input" onSubmit={(event) => { event.preventDefault(); sendChat(); }}>
            <input value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Ask about hooks, metrics, creators, or improvements..." disabled={!analysis} />
            <button type="submit" disabled={!canChat || !message.trim()} aria-label="Send message">
              {isStreaming ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
            </button>
          </form>
        </aside>
      </section>
    </main>
  );
}
