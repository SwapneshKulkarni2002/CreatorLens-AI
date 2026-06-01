# Loom Demo Script

1. Start the backend on `http://localhost:8000` and the frontend on `http://localhost:3000`.
2. Show `.env` has `OPENAI_API_KEY` and `ALLOW_WHISPER_FALLBACK=true`.
3. Paste one YouTube URL and one Instagram Reel URL.
4. Click Analyze and show:
   - metadata cards for Video A and Video B
   - views, likes, comments, creator, follower count when available
   - computed engagement rate
   - indexed transcript chunk count
5. Ask these questions in the chat:
   - Why did Video A get more engagement than Video B?
   - What's the engagement rate of each?
   - Compare the hooks in the first 5 seconds.
   - Who's the creator of Video B and what's their follower count?
   - Suggest improvements for B based on what worked in A.
6. Point out that answers stream, keep prior chat context, and cite transcript chunks.
7. Explain scale:
   - metadata/transcripts are cached by URL in production
   - only transcript chunks are embedded and stored
   - small embedding model plus `gpt-4o-mini` keeps cost low
   - Qdrant local mode can move directly to Qdrant Cloud for managed production scale
   - jobs should run in a queue for 1000 creators/day

Submission reply:

Project URL: add deployed or local demo URL  
Project Description: Full-stack RAG chatbot that compares YouTube and Instagram Reel performance using transcripts, metadata, embeddings, Qdrant vector search, and streaming cited answers.  
Loom URL: add recorded Loom URL  
Github repo: add pushed repository URL
