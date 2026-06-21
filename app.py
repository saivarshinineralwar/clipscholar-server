import os
import re
import uuid
import subprocess
 
from flask import Flask, request, jsonify, send_from_directory
from pytubefix import YouTube
from pytubefix.cli import on_progress
import imageio_ffmpeg
 
app = Flask(__name__)
 
CLIPS_DIR = "clips"
os.makedirs(CLIPS_DIR, exist_ok=True)
 
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
 
 
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
 
    # Pattern 1: "Topic title: X, From time: Y, To time: Z"
    pattern1 = r"Topic title:\s*(.+?),\s*From time:\s*([\d:]+),\s*To time:\s*([\d:]+)"
    matches = re.findall(pattern1, text)
    if matches:
        for title, from_str, to_str in matches:
            segments.append({"title": title.strip(), "from": from_str.strip(), "to": to_str.strip()})
        return segments
 
    # Pattern 2: "1, Topic: X, From: Y, To: Z, Summary: ..."
    pattern2 = r"\d+,\s*Topic:\s*(.+?),\s*From:\s*([\d:]+),\s*To:\s*([\d:]+)"
    matches = re.findall(pattern2, text)
    if matches:
        for title, from_str, to_str in matches:
            segments.append({"title": title.strip(), "from": from_str.strip(), "to": to_str.strip()})
        return segments
 
    # Pattern 3: "1, Title, from_time, to_time, Summary"
    pattern3 = r"\d+,\s*(.+?),\s*([\d:]+),\s*([\d:]+)"
    matches = re.findall(pattern3, text)
    if matches:
        for title, from_str, to_str in matches:
            segments.append({"title": title.strip(), "from": from_str.strip(), "to": to_str.strip()})
        return segments
 
    return segments
 
 
def download_video(youtube_url, output_path):
    yt = YouTube(youtube_url, on_progress_callback=on_progress, use_oauth=False, allow_oauth_cache=False)
    stream = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )
    if not stream:
        raise Exception("No suitable video stream found")
    tmp_dir = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    stream.download(output_path=tmp_dir, filename=filename)
    return output_path
 
 
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "ClipScholar video cutter is running"})
 
 
@app.route("/process", methods=["GET", "POST"])
def process_video():
    youtube_url = (
        request.args.get("youtube_url") or
        request.form.get("youtube_url") or
        (request.get_json(silent=True) or {}).get("youtube_url", "")
    ).strip()
 
    segments_text = (
        request.args.get("segments_text") or
        request.form.get("segments_text") or
        (request.get_json(silent=True) or {}).get("segments_text") or
        request.get_data(as_text=True)
    ).strip()
 
    if not youtube_url:
        return jsonify({"error": "youtube_url is required"}), 400
 
    segments = parse_segments_text(segments_text)
 
    if not segments:
        return jsonify({"error": "Could not parse segments", "preview": segments_text[:300]}), 400
 
    job_id = uuid.uuid4().hex[:10]
    source_path = os.path.join("/tmp", f"{job_id}_source.mp4")
 
    try:
        download_video(youtube_url, source_path)
    except Exception as e:
        return jsonify({"error": f"Failed to download video: {str(e)}"}), 500
 
    if not os.path.exists(source_path):
        return jsonify({"error": "Downloaded file not found"}), 500
 
    clips = []
    for i, seg in enumerate(segments):
        title = seg.get("title", f"Segment {i + 1}")
        start = time_to_seconds(seg.get("from", 0))
        end = time_to_seconds(seg.get("to", start + 1))
        duration = max(end - start, 0.5)
 
        clip_filename = f"{job_id}_{i + 1}_{safe_filename(title)}.mp4"
        clip_path = os.path.join(CLIPS_DIR, clip_filename)
 
        cmd = [
            FFMPEG_PATH, "-y",
            "-ss", str(start),
            "-i", source_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "veryfast",
            clip_path,
        ]
 
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
 
        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
            clip_url = request.host_url.rstrip("/") + f"/clips/{clip_filename}"
            clips.append({"title": title, "from": seg.get("from"), "to": seg.get("to"), "url": clip_url})
        else:
            clips.append({"title": title, "error": "ffmpeg failed", "details": result.stderr[-200:]})
 
    try:
        os.remove(source_path)
    except OSError:
        pass
 
    if not any("url" in c for c in clips):
        return jsonify({"error": "No clips could be generated", "details": clips}), 500
 
    return jsonify({"job_id": job_id, "clips": clips})
 
 
@app.route("/clips/<filename>", methods=["GET"])
def serve_clip(filename):
    return send_from_directory(CLIPS_DIR, filename, as_attachment=False)
 
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
