
import os
import sys
import glob
import json
import uuid
import shutil
import subprocess
from fractions import Fraction
from urllib.parse import urlparse

import boto3
import runpod


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

WORK_ROOT = "/tmp"
OUTPUT_WIDTH = 7680
OUTPUT_HEIGHT = 4320

REALESRGAN_BIN = "/opt/realesrgan/realesrgan-ncnn-vulkan"
REALESRGAN_MODEL = "realesr-animevideov3"

DOWNLOAD_TIMEOUT = 180
MAX_ERROR_CHARS = 8000


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def run_command(command, stage, timeout=None):
    """Run a command and preserve useful stderr when it fails."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{stage} timed out.\n{stderr[-MAX_ERROR_CHARS:]}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{stage}: executable not found: {command[0]}"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"{stage} failed (exit code {result.returncode}).\n"
            f"Command: {command[0]}\n"
            f"STDERR:\n{result.stderr[-MAX_ERROR_CHARS:]}\n"
            f"STDOUT:\n{result.stdout[-2000:]}"
        )

    return result


def validate_url(url):
    """Basic URL validation. This is not a substitute for SSRF protection."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("input.video_url must be a non-empty URL.")

    parsed = urlparse(url.strip())

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("video_url must be a valid HTTP or HTTPS URL.")

    return url.strip()


def download_video(video_url, download_dir):
    """
    Download a YouTube video or direct media URL using yt-dlp.
    Requires yt-dlp and FFmpeg to be installed in the image.
    """
    os.makedirs(download_dir, exist_ok=True)

    output_template = os.path.join(download_dir, "source.%(ext)s")

    command = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--no-part",
        "--socket-timeout", "30",
        "--retries", "3",
        "--fragment-retries", "3",
        "-f", "bv*+ba/b",
        "--merge-output-format", "mp4",
        "--recode-video", "mp4",
        "-o", output_template,
        video_url,
    ]

    run_command(
        command,
        "Downloading/converting source video with yt-dlp",
        timeout=DOWNLOAD_TIMEOUT * 10,
    )

    # Prefer the final MP4 produced by yt-dlp.
    input_path = os.path.join(download_dir, "source.mp4")

    if not os.path.isfile(input_path):
        candidates = [
            path for path in glob.glob(
                os.path.join(download_dir, "source.*")
            )
            if not path.endswith((".part", ".ytdl", ".json"))
        ]

        candidates = [
            path for path in candidates
            if os.path.isfile(path) and os.path.getsize(path) > 0
        ]

        if len(candidates) == 1:
            input_path = candidates[0]
        else:
            raise RuntimeError(
                "yt-dlp completed, but the final video file could not "
                f"be identified. Files found: {candidates}"
            )

    if os.path.getsize(input_path) == 0:
        raise ValueError("Downloaded video is empty.")

    return input_path


def probe_video(input_path):
    """Validate the file and read video metadata using ffprobe."""
    result = run_command(
        [
            "ffprobe",
            "-v", "error",
            "-show_streams",
            "-show_format",
            "-of", "json",
            input_path,
        ],
        "Inspecting downloaded video with ffprobe",
        timeout=60,
    )

    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON.") from exc

    video_streams = [
        stream for stream in metadata.get("streams", [])
        if stream.get("codec_type") == "video"
    ]

    if not video_streams:
        raise ValueError(
            "Downloaded file contains no readable video stream. "
            "It may be an HTML page, an incomplete download, or an "
            "unsupported file."
        )

    stream = video_streams[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")

    try:
        fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError, TypeError):
        fps = 30.0

    if not (1 <= fps <= 120):
        fps = 30.0

    return metadata, fps


def get_r2_client():
    """Create the R2/S3 client only when an upload is needed."""
    endpoint = os.environ.get("R2_ENDPOINT_URL")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET_NAME")

    missing = [
        name for name, value in [
            ("R2_ENDPOINT_URL", endpoint),
            ("R2_ACCESS_KEY_ID", access_key),
            ("R2_SECRET_ACCESS_KEY", secret_key),
            ("R2_BUCKET_NAME", bucket),
        ]
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Missing required storage environment variables: "
            + ", ".join(missing)
        )

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    return client, bucket


# ---------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------

def upscale_video_to_8k(event):
    job_id = event.get("id", uuid.uuid4().hex)
    job_input = event.get("input") or {}

    input_video_url = job_input.get("video_url")
    requested_fps = job_input.get("fps")

    if not input_video_url:
        return {
            "status": "failed",
            "error": "Missing video_url in input payload.",
        }

    try:
        input_video_url = validate_url(input_video_url)

        if requested_fps is not None:
            requested_fps = float(requested_fps)
            if not (1 <= requested_fps <= 120):
                raise ValueError("fps must be between 1 and 120.")

        work_dir = os.path.join(WORK_ROOT, str(job_id))
        download_dir = os.path.join(work_dir, "source")
        frames_in = os.path.join(work_dir, "frames_in")
        frames_out = os.path.join(work_dir, "frames_out")
        output_path = os.path.join(work_dir, "output_8k.mp4")

        os.makedirs(frames_in, exist_ok=True)
        os.makedirs(frames_out, exist_ok=True)

        try:
            # 1. Download a real video file.
            runpod.serverless.progress_update(
                event, "Downloading source video..."
            )

            input_path = download_video(
                input_video_url, download_dir
            )

            # 2. Verify the downloaded file and discover its FPS.
            runpod.serverless.progress_update(
                event, "Validating source video..."
            )

            metadata, source_fps = probe_video(input_path)
            target_fps = requested_fps or source_fps

            # 3. Extract frames and capture FFmpeg's real error.
            runpod.serverless.progress_update(
                event, "Extracting video frames..."
            )

            extract_command = [
                "ffmpeg", "-hide_banner", "-y",
                "-i", input_path,
                "-map", "0:v:0",
                "-vsync", "0",
            ]

            # If the caller explicitly requests an FPS, normalize
            # the extracted frames to that frame rate.
            if requested_fps is not None:
                extract_command.extend([
                    "-vf", f"fps={target_fps}",
                ])

            extract_command.extend([
                "-qscale:v", "2",
                os.path.join(frames_in, "frame_%06d.jpg"),
            ])

            run_command(
                extract_command,
                "Extracting video frames with FFmpeg",
                timeout=600,
            )

            input_frames = glob.glob(
                os.path.join(frames_in, "frame_*.jpg")
            )

            if not input_frames:
                raise RuntimeError(
                    "FFmpeg returned successfully but extracted "
                    "zero frames."
                )

            # 4. Upscale extracted frames using Real-ESRGAN.
            runpod.serverless.progress_update(
                event, "Upscaling frames on GPU..."
            )

            run_command(
                [
                    REALESRGAN_BIN,
                    "-i", frames_in,
                    "-o", frames_out,
                    "-n", REALESRGAN_MODEL,
                    "-s", "4",
                    "-t", "512",
                    "-f", "jpg",
                ],
                "Real-ESRGAN frame upscaling",
                timeout=7200,
            )

            output_frames = glob.glob(
                os.path.join(frames_out, "frame_*.jpg")
            )

            if len(output_frames) != len(input_frames):
                raise RuntimeError(
                    "Frame count mismatch: "
                    f"{len(input_frames)} input frames, "
                    f"{len(output_frames)} upscaled frames."
                )

            # 5. Encode an 8K HEVC video. Preserve aspect ratio by
            # fitting the image inside the 7680x4320 canvas.
            runpod.serverless.progress_update(
                event, "Encoding 8K video..."
            )

            frame_pattern = os.path.join(
                frames_out, "frame_%06d.jpg"
            )

            common_command = [
                "ffmpeg", "-hide_banner", "-y",
                "-framerate", str(target_fps),
                "-i", frame_pattern,
                "-i", input_path,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-vf",
                (
                    f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:"
                    "force_original_aspect_ratio=decrease:"
                    "flags=lanczos,"
                    f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:"
                    "(ow-iw)/2:(oh-ih)/2"
                ),
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
            ]

            # Prefer NVIDIA hardware encoding; fall back to x265
            # if NVENC is unavailable or fails.
            nvenc_command = common_command + [
                "-c:v", "hevc_nvenc",
                "-preset", "p5",
                "-cq", "21",
                "-b:v", "0",
                output_path,
            ]

            try:
                run_command(
                    nvenc_command,
                    "Encoding video with NVIDIA NVENC",
                    timeout=7200,
                )
            except RuntimeError as nvenc_error:
                if os.path.exists(output_path):
                    os.remove(output_path)

                print(
                    "NVENC failed; trying CPU x265 encoder. "
                    f"Original error: {nvenc_error}"
                )

                x265_command = common_command + [
                    "-c:v", "libx265",
                    "-preset", "medium",
                    "-crf", "23",
                    output_path,
                ]

                run_command(
                    x265_command,
                    "Encoding video with libx265 fallback",
                    timeout=14400,
                )

            if (
                not os.path.isfile(output_path)
                or os.path.getsize(output_path) == 0
            ):
                raise RuntimeError(
                    "Encoding finished without a valid output file."
                )

            # 6. Upload the result to Cloudflare R2.
            runpod.serverless.progress_update(
                event, "Uploading completed video..."
            )

            s3_client, bucket = get_r2_client()

            output_key = (
                f"upscaled_8k/{job_id}_"
                f"{uuid.uuid4().hex[:6]}.mp4"
            )

            s3_client.upload_file(
                output_path, bucket, output_key
            )

            presigned_url = s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": bucket,
                    "Key": output_key,
                },
                ExpiresIn=86400,
            )

            return {
                "status": "completed",
                "output_8k_url": presigned_url,
                "storage_key": output_key,
                "source_fps": source_fps,
                "output_fps": target_fps,
                "input_frames": len(input_frames),
                "output_frames": len(output_frames),
                "output_resolution": "7680x4320",
            }

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    except Exception as exc:
        # Log the full error to the RunPod worker logs.
        import traceback
        traceback.print_exc()

        # Raise instead of returning a success-shaped response.
        raise RuntimeError(
            f"Video upscaling failed: {type(exc).__name__}: {exc}"
        ) from exc


runpod.serverless.start({"handler": upscale_video_to_8k})
