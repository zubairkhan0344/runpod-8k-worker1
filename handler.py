import os
import uuid
import shutil
import subprocess
import boto3
import requests
import runpod

# Initialize S3/R2 client globally (outside handler) so warm workers don't reconnect
s3_client = boto3.client(
    "s3",
    endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
    aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
    region_name="auto",
)
BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")


def upscale_video_to_8k(event):
    job_id = event["id"]
    job_input = event["input"]

    input_video_url = job_input.get("video_url")
    target_fps = job_input.get("fps", 30)

    if not input_video_url:
        return {"error": "Missing video_url in input payload."}

    work_dir = f"/tmp/{job_id}"
    frames_in = f"{work_dir}/frames_in"
    frames_out = f"{work_dir}/frames_out"
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    input_path = f"{work_dir}/input.mp4"
    output_path = f"{work_dir}/output_8k.mp4"

    try:
        # 1. Download source video from your website/storage
        runpod.serverless.progress_update(event, "Downloading source video...")
        with requests.get(input_video_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(input_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # 2. Extract frames using FFmpeg
        runpod.serverless.progress_update(event, "Extracting video frames...")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-qscale:v", "2", f"{frames_in}/frame_%06d.jpg"
        ], check=True)

        # 3. Upscale frames with Real-ESRGAN (Tiled for 24GB-32GB VRAM safety)
        runpod.serverless.progress_update(event, "Upscaling frames to 8K on GPU...")
        subprocess.run([
            "/opt/realesrgan/realesrgan-ncnn-vulkan",
            "-i", frames_in,
            "-o", frames_out,
            "-n", "realesr-animevideov3",
            "-s", "4",
            "-t", "512",
            "-f", "jpg"
        ], check=True)

        # 4. Stitch 8K frames + original audio using NVIDIA NVENC hardware encoder
        runpod.serverless.progress_update(event, "Encoding 8K HEVC video via NVENC...")
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(target_fps),
            "-i", f"{frames_out}/frame_%06d.jpg",
            "-i", input_path,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-vf", "scale=7680:4320:flags=lanczos",
            "-c:v", "hevc_nvenc", "-preset", "p5", "-cq", "21",
            "-pix_fmt", "yuv420p", "-c:a", "copy",
            output_path
        ], check=True)

        # 5. Upload finished 8K MP4 to Cloudflare R2 / S3 and generate download URL
        runpod.serverless.progress_update(event, "Uploading 8K video to storage...")
        output_key = f"upscaled_8k/{job_id}_{uuid.uuid4().hex[:6]}.mp4"
        s3_client.upload_file(output_path, BUCKET_NAME, output_key)

        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET_NAME, "Key": output_key},
            ExpiresIn=86400
        )

        return {
            "status": "completed",
            "output_8k_url": presigned_url,
            "storage_key": output_key
        }

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": upscale_video_to_8k})
