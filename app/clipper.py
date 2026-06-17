"""
Cuts clips out of the source video using FFmpeg, based on ranked moments.
"""

import os
import subprocess

from app.models import ClipResult, RankedMoment


def get_video_duration(video_path: str) -> float:
    """Return the duration of `video_path` in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def cut_clips(
    video_path: str,
    moments: list[RankedMoment],
    output_dir: str,
    job_id: str,
    base_url: str = "",
) -> list[ClipResult]:
    """
    For each moment, cut a clip from `video_path` into `output_dir`.

    Timestamps from the LLM are clamped to the video's actual duration
    (the LLM occasionally returns slightly out-of-range values). Moments
    that don't yield any real overlap with the video are skipped.

    Clips are always re-encoded rather than stream-copied. `-ss` before
    `-i` does a fast input seek to the nearest keyframe, but re-encoding
    additionally trims the output to the exact requested start/end time -
    stream copy alone would snap to the keyframe and produce inaccurate cuts.

    `base_url` is used to build the `download_url` field in the response
    (e.g. "https://your-deployment.example.com"). If empty, the URL is relative.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_duration = get_video_duration(video_path)

    results: list[ClipResult] = []

    for i, moment in enumerate(moments):
        start = max(0.0, min(moment.start, video_duration))
        end = max(0.0, min(moment.end, video_duration))

        if end <= start:
            # Moment falls entirely outside the video, or start/end were
            # reversed/equal after clamping - skip it.
            continue

        duration = end - start
        clip_filename = f"{job_id}_clip{i + 1}.mp4"
        clip_path = os.path.join(output_dir, clip_filename)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            clip_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
            raise RuntimeError(f"FFmpeg produced an empty clip for moment {i + 1} (job {job_id})")

        download_url = f"{base_url.rstrip('/')}/clips/{clip_filename}"

        results.append(ClipResult(
            clip_filename=clip_filename,
            download_url=download_url,
            start=start,
            end=end,
            duration=duration,
            reason=moment.reason,
            score=moment.score,
        ))

    return results
