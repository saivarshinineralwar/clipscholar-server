import os
import re
import uuid
import subprocess
import threading
import time

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import yt_dlp
import imageio_ffmpeg
import assemblyai as aai

# Load environment variables from .env file
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

load_env()

ASSEMBLYAI_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
aai.settings.api_key = ASSEMBLYAI_KEY

app = Flask(__name__)
CORS(app)

CLIPS_DIR = "clips"
os.makedirs(CLIPS_DIR, exist_ok=True)

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIES_FILE = os.path.join(BASE_DIR, "cookies.txt")

jobs = {}


def time_to_seconds(t):
    if isinstance(t, (int, float)):
        return float(t)
    t = str(t).strip()
    parts = [float(p) for p in t.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    return parts[0]


def safe_filename(text, max_len=40):
    text = re.sub(r"[^a-zA-Z0-9_\- ]", "", text or "")
    text = text.strip().replace(" ", "_")
    return text[:max_len] if text else "clip"


def seconds_to_time(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"


def transcribe_video(youtube_url):
    """Transcribe YouTube video using AssemblyAI."""
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(youtube_url)
    if transcript.status == aai.TranscriptStatus.error:
        raise Exception(f"Transcription error: {transcript.error}")
    return transcript.text


def segment_transcript(transcript_text, duration_hint=None):
    """Use AssemblyAI LeMUR to segment transcript into topics."""
    client = aai.Client()
    
    prompt = """You are a video segmentation tool. Split this transcript into topic segments.

Output ONLY this exact format for each segment, one per line:
SEGMENT|Topic Title Here|0:00|1:30

Rules:
- Start each line with SEGMENT|
- Use pipe | to separate the 4 fields
- Times in M:SS format
- Cover 100% of the video content
- No extra text, no summaries, no numbering outside this format

Transcript:
""" + transcript_text

    # Use AssemblyAI LeMUR for AI segmentation
    result = client.lemur.task(
        prompt=prompt,
        final_model=aai.LemurModel.claude3_haiku,
        input_text=transcript_text,
        max_output_size=2000
    )
    return result.response


def parse_segments(text):
    segments = []
    pattern = r"SEGMENT\|(.+?)\|([\d:]+)\|([\d:]+)"
    matches = re.findall(pattern, text)
    for title, from_str, to_str in matches:
        segments.append({
            "title": title.strip(),
            "from": from_str.strip(),
            "to": to_str.strip()
        })
    return segments


def process_job(job_id, youtube_url, base_url):
    """Full pipeline: transcribe → segment → download → cut clips."""
    try:
        # Step 1: Transcribe
        jobs[job_id]["status"] = "transcribing"
        jobs[job_id]["message"] = "Transcribing audio..."
        transcript = transcribe_video(youtube_url)
        
        if not transcript:
            jobs[job_id] = {"status": "error", "error": "Transcription returned empty"}
            return

        # Step 2: Segment with AI
        jobs[job_id]["status"] = "segmenting"
        jobs[job_id]["message"] = "AI is analyzing topics..."
        segments_text = segment_transcript(transcript)
        segments = parse_segments(segments_text)

        if not segments:
            jobs[job_id] = {"status": "error", "error": "Could not parse segments from AI output", "raw": segments_text[:300]}
            return

        # Step 3: Download video
        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["message"] = f"Downloading video ({len(segments)} clips to create)..."
        
        source_path = os.path.join("/tmp", f"{job_id}_source.mp4")
        ydl_opts = {
            "format": "18/best[ext=mp4][height<=480]/best[height<=480]/best",
            "outtmpl": source_path,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
        }
        if os.path.exists(COOKIES_FILE):
            ydl_opts["cookiefile"] = COOKIES_FILE

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        if not os.path.exists(source_path):
            jobs[job_id] = {"status": "error", "error": "Video download failed"}
            return

        # Step 4: Cut clips
        jobs[job_id]["status"] = "cutting"
        jobs[job_id]["message"] = "Cutting clips..."
        clips = []

        for i, seg in enumerate(segments):
            title = seg["title"]
            start = time_to_seconds(seg["from"])
            end = time_to_seconds(seg["to"])
            duration = max(end - start, 0.5)

            clip_filename = f"{job_id}_{i+1}_{safe_filename(title)}.mp4"
            clip_path = os.path.join(CLIPS_DIR, clip_filename)

            cmd = [FFMPEG_PATH, "-y", "-ss", str(start), "-i", source_path,
                   "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
                   "-preset", "veryfast", clip_path]
            subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                clips.append({
                    "title": title,
                    "from": seg["from"],
                    "to": seg["to"],
                    "url": f"{base_url}/clips/{clip_filename}"
                })

        try:
            os.remove(source_path)
        except OSError:
            pass

        jobs[job_id] = {"status": "done", "clips": clips}

    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "cookies": "loaded" if os.path.exists(COOKIES_FILE) else "missing",
        "assemblyai": "configured" if ASSEMBLYAI_KEY else "missing"
    })


@app.route("/start", methods=["GET", "POST"])
def start():
    """Start full pipeline — returns job_id immediately."""
    youtube_url = (
        request.args.get("youtube_url") or
        request.form.get("youtube_url") or
        (request.get_json(silent=True) or {}).get("youtube_url", "")
    ).strip()

    if not youtube_url:
        return jsonify({"error": "youtube_url is required"}), 400

    job_id = uuid.uuid4().hex[:10]
    base_url = request.host_url.rstrip("/")
    jobs[job_id] = {"status": "queued", "message": "Starting..."}

    threading.Thread(
        target=process_job,
        args=(job_id, youtube_url, base_url),
        daemon=True
    ).start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/status/<job_id>", methods=["GET"])
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/clips/<filename>", methods=["GET"])
def serve_clip(filename):
    return send_from_directory(CLIPS_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting ClipScholar server on port {port}")
    print(f"AssemblyAI: {'configured' if ASSEMBLYAI_KEY else 'MISSING - check .env file'}")
    print(f"Cookies: {'loaded' if os.path.exists(COOKIES_FILE) else 'missing'}")
    app.run(host="0.0.0.0", port=port, threaded=True)
