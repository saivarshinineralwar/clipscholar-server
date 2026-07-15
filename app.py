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


def parse_segments(text):
    segments = []
    # Primary: SEGMENT|Title|0:00|1:30
    pattern1 = r"SEGMENT\|(.+?)\|([\d:]+)\|([\d:]+)"
    matches = re.findall(pattern1, text)
    if matches:
        for title, from_str, to_str in matches:
            segments.append({"title": title.strip(), "from": from_str.strip(), "to": to_str.strip()})
        return segments
    # Fallback patterns
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


def process_job(job_id, youtube_url, base_url):
    try:
        # Step 1: Download audio for transcription
        jobs[job_id] = {"status": "downloading_audio", "message": "Downloading audio for transcription..."}
        
        audio_path = os.path.join("/tmp", f"{job_id}_audio.mp3")
        ydl_audio_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path.replace(".mp3", ""),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }],
            "quiet": True,
            "no_warnings": True,
        }
        if os.path.exists(COOKIES_FILE):
            ydl_audio_opts["cookiefile"] = COOKIES_FILE

        with yt_dlp.YoutubeDL(ydl_audio_opts) as ydl:
            ydl.download([youtube_url])

        # Find the downloaded audio file
        actual_audio = None
        for f in os.listdir("/tmp"):
            if f.startswith(f"{job_id}_audio") and (f.endswith(".mp3") or f.endswith(".m4a") or f.endswith(".webm")):
                actual_audio = os.path.join("/tmp", f)
                break

        if not actual_audio or not os.path.exists(actual_audio):
            jobs[job_id] = {"status": "error", "error": "Audio download failed"}
            return

        # Step 2: Transcribe with AssemblyAI
        jobs[job_id] = {"status": "transcribing", "message": "Transcribing audio with AI..."}
        
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(actual_audio)
        
        try:
            os.remove(actual_audio)
        except OSError:
            pass

        if transcript.status == aai.TranscriptStatus.error:
            jobs[job_id] = {"status": "error", "error": f"Transcription failed: {transcript.error}"}
            return

        transcript_text = transcript.text
        if not transcript_text:
            jobs[job_id] = {"status": "error", "error": "Transcript is empty"}
            return

        # Step 3: Segment with AI using LeMUR
        jobs[job_id] = {"status": "segmenting", "message": "AI is analyzing and splitting into topics..."}
        
        prompt = """You are a video segmentation tool. Split this transcript into topic segments.

Output ONLY this exact format for each segment, one per line, nothing else:
SEGMENT|Topic Title Here|0:00|1:30

Rules:
- Start each line with SEGMENT|
- Use pipe | to separate the 4 fields: SEGMENT, title, start time, end time
- Times in M:SS format
- Cover 100% of the video content with no gaps
- No extra text, no numbering, no summaries"""

        result = transcript.lemur.task(
            prompt=prompt,
            final_model=aai.LemurModel.claude3_haiku,
            max_output_size=2000
        )
        
        segments = parse_segments(result.response)

        if not segments:
            jobs[job_id] = {"status": "error", "error": "Could not parse segments", "raw": result.response[:300]}
            return

        # Step 4: Download video for cutting
        jobs[job_id] = {"status": "downloading_video", "message": f"Downloading video to cut {len(segments)} clips..."}
        
        source_path = os.path.join("/tmp", f"{job_id}_source.mp4")
        ydl_opts = {
            "format": "18/best[ext=mp4][height<=480]/best[height<=480]/best",
            "outtmpl": source_path,
            "quiet": True,
            "no_warn
