# ClipScholar Video Cutter

A tiny server that takes a YouTube link plus AI-generated topic segments,
downloads the video (yt-dlp), cuts one clip per topic (ffmpeg), and returns
download links for each clip.

## Endpoint

`POST /process`

Body (send EITHER `segments_text` raw from the AI, OR a structured `segments` list):

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "segments_text": "Segment 1, Topic title: Introduction, From time: 0:00, To time: 1:30, Summary: ...\nSegment 2, Topic title: Supply and Demand, From time: 1:30, To time: 5:45, Summary: ..."
}
```

Response:

```json
{
  "job_id": "abc123",
  "clips": [
    {"title": "Introduction", "from": "0:00", "to": "1:30", "url": "https://yourserver.onrender.com/clips/abc123_1_Introduction.mp4"},
    {"title": "Supply and Demand", "from": "1:30", "to": "5:45", "url": "https://yourserver.onrender.com/clips/abc123_2_Supply_and_Demand.mp4"}
  ]
}
```

## Deploying on Render (free tier)

1. Push this folder to a new GitHub repo.
2. On Render.com, click "New +" -> "Web Service" -> connect the repo.
3. Environment: Python 3
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 600`
6. Deploy. Render gives you a live URL like `https://clipscholar-server.onrender.com`

Note: the free tier "spins down" after inactivity and takes ~30-60 seconds
to wake up on the first request after idling. This is fine for a
"Processing..." status flow.
