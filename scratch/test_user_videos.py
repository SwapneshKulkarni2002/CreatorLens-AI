import sys
sys.path.insert(0, r"C:/Users/hp/Documents/GitHub/CreatorLens-AI/backend")

import httpx
from app.extractors import extract_video
from app.rag import _embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

yt_url = "https://youtu.be/-BEknhlLFsE?si=X7IoxCqYwICk0VXR"
insta_url = "https://www.instagram.com/fallontonight/reel/C20i65_Pu0k/?hl=en"

print("--- Testing Instagram extraction ---")
try:
    insta_res = extract_video(insta_url, "B", "instagram")
    print("Insta extraction metadata:", insta_res.metadata)
    print("Insta transcript length:", len(insta_res.transcript))
except Exception as e:
    print("Insta extraction failed:", e)

print("\n--- Testing YouTube extraction & embedding ---")
try:
    yt_res = extract_video(yt_url, "A", "youtube")
    print("YouTube extraction title:", yt_res.metadata.get("title"))
    print("YouTube transcript length:", len(yt_res.transcript))
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_text(yt_res.transcript)
    print(f"Split into {len(chunks)} chunks.")
    
    emb = _embeddings()
    for i, chunk in enumerate(chunks, start=1):
        try:
            print(f"Embedding chunk {i} (len: {len(chunk)})...")
            vec = emb.embed_query(chunk)
            print(f"Chunk {i} embedding success!")
        except Exception as e:
            print(f"FAILED chunk {i}: {e}")
            if hasattr(e, "response") and hasattr(e.response, "text"):
                print("Response detail:", e.response.text)
except Exception as e:
    print("YouTube extraction/embedding failed:", e)
