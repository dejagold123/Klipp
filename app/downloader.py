"""
Downloads source video using yt-dlp.
"""

import os
from typing import Optional

import yt_dlp


def download_video(video_url: str, output_dir: str, job_id: str) -> tuple[str, Optional[str]]:
    """
    Download `video_url` into `output_dir`, returning (local_file_path, video_title).

    The output filename is based on `job_id` so concurrent jobs don't clash.
    `video_title` is whatever yt-dlp extracted; it's None if extraction
    didn't surface a title (e.g. for a direct video file URL).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, f"{job_id}.%(ext)s")

    ydl_opts = {
        "format": "mp4/bestvideo+bestaudio/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        # extract_info with download=True returns the info dict;
        # prepare_filename gives the actual path used (post merge it's .mp4)
        filename = ydl.prepare_filename(info)

    title = info.get("title") if isinstance(info, dict) else None

    # If merged to mp4 but prepare_filename returned a different extension,
    # fall back to checking for the expected mp4 path.
    mp4_path = os.path.join(output_dir, f"{job_id}.mp4")
    if os.path.exists(mp4_path):
        return mp4_path, title
    if os.path.exists(filename):
        return filename, title

    raise FileNotFoundError(f"Downloaded file not found for job {job_id}")
