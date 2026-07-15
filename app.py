import os
import re
import uuid
import subprocess
import threading

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import yt_dlp
import imageio_ffmpeg
import assemblyai as aai
from groq import Groq

def load_env():
    for name in [".env", "config.env"]:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip()
            break

load_env()

ASSEMBLYAI_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

aai.settings.api_key = ASSEMBLYAI_KEY

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
FFMPEG_DIR = os.path.dirname(FFMPEG_PATH)
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

app = Flask(__name__)
CORS(app)

CLIPS_DIR = "clips"
os.makedirs(CLIPS_DIR, exist_ok=True)

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


def parse_segments(text):
    segments = []
    pattern1 = r"SEGMENT\|(.+?)\|([\d:]+)\|([\d:]+)"
    matches = re.findall(pattern1, text)
    if matches:
        for title, from_str, to_str in matches:
            segments.append({"title": title.strip(), "from": from_str.strip(), "to": to_str.strip()})
        return segments
    pattern2 = r"Topic title:\s*(.+?),\s*From time:\s*([\d:]+),\s*To time:\s*([\d:]+)"
    matches = re.findall(pattern2, text)
    if matches:
        for title, from_str, to_str in matches:
            segments.append({"title": title.strip(), "from": from_str.strip(), "to": to_str.strip()})
        return segments
    pattern3 = r"\d+,\s*Topic:\s*(.+?),\s*From:\s*([\d:]+),\s*To:\s*([\d:]+)"
    matches = re.findall(pattern3, text)
    if matches:
        for title, from_str, to_str in matches:
            segments.append({"title": title.strip(), "from": from_str.strip(), "to": to_str.strip()})
        return segments
    return segments


def segment_with_groq(transcript_text):
    client = Groq(api_key=GROQ_KEY)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are a video segmentation tool. Output ONLY the segment lines, nothing else."
            },
            {
                "role": "user",
                "content": f"""Split this transcript into topic segments. Output ONLY this exact format, one per line:
SEGMENT|Topic Title|0:00|1:30

Rules:
- Start each line with SEGMENT|
- Use pipe | to separate: SEGMENT, title, start time, end time
- Times in M:SS format
- Cover 100% of content with no gaps
- No extra text at all

Transcript:
{transcript_text[:6000]}"""
            }
        ],
        max_tokens=2000,
        temperature=0.1
    )
    return response.choices[0].message.content


def process_job(job_id, youtube_url, base_url):
    try:
        # Step 1: Download video
        jobs[job_id] = {"status": "downloading", "message": "Downloading video..."}

        source_path = os.path.join(os.environ.get("TEMP", "C:/Temp"), f"{job_id}_source.mp4")
        os.makedirs(os.path.dirname(source_path), exist_ok=True)

        ydl_opts = {
            "format": "18/best[ext=mp4][height<=480]/best[height<=480]/best",
            "outtmpl": source_path,
            "quiet": True,
            "no_warnings": True,
            "ffmpeg_location": FFMPEG_DIR,
        }
        if os.path.exists(COOKIES_FILE):
            ydl_opts["cookiefile"] = COOKIES_FILE

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        if not os.path.exists(source_path):
            jobs[job_id] = {"status": "error", "error": "Video download failed"}
            return

        # Step 2: Extract audio
        jobs[job_id] = {"status": "transcribing", "message": "Extracting audio..."}
        audio_path = source_path.replace("_source.mp4", "_audio.mp3")
        cmd = [FFMPEG_PATH, "-y", "-i", source_path, "-vn", "-acodec", "mp3", "-ab", "128k", audio_path]
        subprocess.run(cmd, capture_output=True, timeout=120)

        if not os.path.exists(audio_path):
            jobs[job_id] = {"status": "error", "error": "Audio extraction failed"}
            return

        # Step 3: Transcribe with AssemblyAI
        jobs[job_id] = {"status": "transcribing", "message": "Transcribing audio with AI..."}
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path)

        try:
            os.remove(audio_path)
        except OSError:
            pass

        if transcript.status == aai.TranscriptStatus.error:
            jobs[job_id] = {"status": "error", "error": f"Transcription failed: {transcript.error}"}
            return

        if not transcript.text:
            jobs[job_id] = {"status": "error", "error": "Transcript is empty"}
            return

        # Step 4: Segment with Groq
        jobs[job_id] = {"status": "segmenting", "message": "AI is splitting into topics..."}
        segments_text = segment_with_groq(transcript.text)
        segments = parse_segments(segments_text)

        if not segments:
            jobs[job_id] = {"status": "error", "error": "Could not parse segments", "raw": segments_text[:300]}
            return

        # Step 5: Cut clips
        jobs[job_id] = {"status": "cutting", "message": f"Cutting {len(segments)} clips..."}
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
        "assemblyai": "configured" if ASSEMBLYAI_KEY else "MISSING",
        "groq": "configured" if GROQ_KEY else "MISSING",
        "cookies": "loaded" if os.path.exists(COOKIES_FILE) else "missing",
        "ffmpeg": FFMPEG_PATH
    })


@app.route("/start", methods=["GET", "POST"])
def start():
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
        return jsonify({"error": "Job 
