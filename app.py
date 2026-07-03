import os
import re
import uuid
import subprocess
import threading
import requests

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import yt_dlp
import imageio_ffmpeg

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


def parse_segments_text(text):
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
    pattern3 = r"\d+,\s*Topic:\s*(.+?),\s*From timestamp:\s*([\d:]+),\s*To timestamp:\s*([\d:]+)"
    matches = re.findall(pattern3, text)
    if matches:
        for title, from_str, to_str in matches:
            segments.append({"title": title.strip(), "from": from_str.strip(), "to": to_str.strip()})
        return segments
    pattern4 = r"\d+,\s*Topic:\s*(.+?),\s*From:\s*([\d:]+),\s*To:\s*([\d:]+)"
    matches = re.findall(pattern4, text)
    if matches:
        for title, from_str, to_str in matches:
            segments.append({"title": title.strip(), "from": from_str.strip(), "to": to_str.strip()})
        return segments
    return segments


def download_and_cut(job_id, youtube_url, segments_text, base_url):
    try:
        jobs[job_id]["status"] = "downloading"
        segments = parse_segments_text(segments_text)
        if not segments:
            jobs[job_id] = {"status": "error", "error": "Could not parse segments"}
            return

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
            jobs[job_id] = {"status": "error", "error": "Download failed"}
            return

        jobs[job_id]["status"] = "cutting"
        clips = []

        for i, seg in enumerate(segments):
            title = seg.get("title", f"Segment {i + 1}")
            start = time_to_seconds(seg.get("from", 0))
            end = time_to_seconds(seg.get("to", start + 1))
            duration = max(end - start, 0.5)
            clip_filename = f"{job_id}_{i + 1}_{safe_filename(title)}.mp4"
            clip_path = os.path.join(CLIPS_DIR, clip_filename)
            cmd = [FFMPEG_PATH, "-y", "-ss", str(start), "-i", source_path,
                   "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
                   "-preset", "veryfast", clip_path]
            subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                clips.append({"title": title, "from": seg.get("from"),
                               "to": seg.get("to"), "url": f"{base_url}/clips/{clip_filename}"})

        try:
            os.remove(source_path)
        except OSError:
            pass

        jobs[job_id] = {"status": "done", "clips": clips}

    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}


@app.route("/", methods=["GET"])
def health():
    cookies_status = "loaded" if os.path.exists(COOKIES_FILE) else "missing"
    return jsonify({"status": "ok", "cookies": cookies_status})


@app.route("/process", methods=["GET", "POST"])
def process_video():
    youtube_url = (request.args.get("youtube_url") or
                   request.form.get("youtube_url") or
                   (request.get_json(silent=True) or {}).get("youtube_url", "")).strip()
    segments_text = (request.args.get("segments_text") or
                     request.form.get("segments_text") or
                     (request.get_json(silent=True) or {}).get("segments_text") or
                     request.get_data(as_text=True)).strip()

    if not youtube_url:
        return jsonify({"error": "youtube_url is required"}), 400

    job_id = uuid.uuid4().hex[:10]
    base_url = request.host_url.rstrip("/")
    jobs[job_id] = {"status": "queued"}
    threading.Thread(target=download_and_cut,
                     args=(job_id, youtube_url, segments_text, base_url),
                     daemon=True).start()
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
    
@app.route("/transcribe", methods=["GET"])
def transcribe():
    youtube_url = request.args.get("youtube_url", "").strip()
    if not youtube_url:
        return jsonify({"error": "youtube_url required"}), 400
    
    MAKE_WEBHOOK = "https://hook.eu1.make.com/tcdmsacq1uyhuerysqwkwfwa4fm7t7gd"
    response = requests.get(f"{MAKE_WEBHOOK}?audio_url={youtube_url}", timeout=300)
    return response.text, response.status_code, {"Content-Type": "application/json"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
