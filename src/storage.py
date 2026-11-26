"""S3-compatible object storage upload + extra-LoRA download.

The upload half mirrors the sibling workers' storage.py (same client
setup, same public/presigned URL logic); the payload is an mp4. The
LoRA half is ported from the MiniMax-H3 worker (worker-lora) and
adapted to Wan 2.2's high/low file pairs: at boot the worker fetches
the env-configured LoRA files itself, so LoRA swaps are an env change
+ restart, never a rebuild.
"""

import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import boto3
from botocore.config import Config as BotoConfig

from src import config
from src.errors import UploadError
from src.workflow import LIGHTX2V_HIGH_NAME, LIGHTX2V_LOW_NAME

logger = logging.getLogger("wan-2.2-mmaudio-comfy")


def _get_s3_client():
    kwargs = {
        "service_name": "s3",
        "region_name": config.S3_REGION,
        "aws_access_key_id": config.S3_ACCESS_KEY_ID,
        "aws_secret_access_key": config.S3_SECRET_ACCESS_KEY,
        "config": BotoConfig(
            connect_timeout=10,
            read_timeout=config.UPLOAD_TIMEOUT,
            retries={"max_attempts": 2},
        ),
    }
    if config.S3_ENDPOINT:
        kwargs["endpoint_url"] = config.S3_ENDPOINT
    return boto3.client(**kwargs)


def upload_video(
    local_path: str,
    job_id: str,
    bucket: str,
    prefix: str,
    public: bool = True,
    ttl_seconds: int = 0,
) -> str:
    """Upload an mp4 to S3-compatible storage. Returns the public or signed URL."""
    if not bucket:
        raise UploadError(
            "No S3 bucket configured.",
            hint="Set S3_BUCKET_DEFAULT or pass output.bucket in the request.",
        )

    now = datetime.now(timezone.utc)
    date_prefix = now.strftime("%Y/%m/%d")
    key = f"{prefix}{date_prefix}/{job_id}.mp4"

    client = _get_s3_client()
    file_size = os.path.getsize(local_path)
    logger.info("Uploading %s (%.1f MB) -> s3://%s/%s", local_path, file_size / (1024 * 1024), bucket, key)

    try:
        client.upload_file(
            Filename=local_path,
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
    except Exception as e:
        raise UploadError(
            f"S3 upload failed: {e}",
            hint="Check S3 credentials, bucket name, and network connectivity.",
        )

    if public and config.S3_PUBLIC_BASE_URL:
        url = f"{config.S3_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    elif not public and ttl_seconds > 0:
        try:
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=ttl_seconds,
            )
        except Exception as e:
            raise UploadError(
                f"Failed to generate presigned URL: {e}",
                hint="Check S3 credentials and permissions.",
            )
    elif config.S3_ENDPOINT:
        url = f"{config.S3_ENDPOINT.rstrip('/')}/{bucket}/{key}"
    else:
        url = f"https://{bucket}.s3.{config.S3_REGION}.amazonaws.com/{key}"

    logger.info("Upload complete: %s", url)
    return url


def _download_one_lora(key: str, dest_dir: str) -> str:
    """Fetch one LoRA object from the default bucket into dest_dir.
    Returns the local filename. Raises UploadError when unfetchable."""
    bucket = config.S3_BUCKET_DEFAULT
    if not bucket:
        raise UploadError(
            "An extra LoRA is configured but no S3 bucket is set.",
            hint="Set S3_BUCKET_DEFAULT so the worker can fetch the LoRA file.",
        )
    name = os.path.basename(key)
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest):
        logger.info("Extra LoRA already present: %s", dest)
        return name
    os.makedirs(dest_dir, exist_ok=True)
    client = _get_s3_client()
    logger.info("Downloading extra LoRA s3://%s/%s -> %s", bucket, key, dest)
    try:
        client.download_file(bucket, key, dest)
    except Exception as e:
        raise UploadError(
            f"Extra LoRA download failed: {e}",
            hint=f"Check the key {key!r} matches the object key in the bucket.",
        )
    logger.info(
        "Extra LoRA ready: %s (%.1f MB)",
        dest, os.path.getsize(dest) / (1024 * 1024),
    )
    return name


def _download_lora_from_url(url: str, dest_dir: str) -> str:
    """Fetch one LoRA over HTTP(S) into dest_dir. Returns the local
    filename (the basename of the URL path). For LoRAs too large for
    the storage web UI's upload limit, the worker pulls the file itself
    at boot. Raises UploadError when unfetchable."""
    name = os.path.basename(urllib.parse.urlparse(url).path)
    if not name:
        raise UploadError(
            f"Extra LoRA URL has no filename: {url!r}",
            hint="Use a URL that ends with the .safetensors filename.",
        )
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest):
        logger.info("Extra LoRA already present: %s", dest)
        return name
    os.makedirs(dest_dir, exist_ok=True)
    tmp = dest + ".part"
    logger.info("Downloading extra LoRA %s -> %s", url, dest)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise UploadError(
            f"Extra LoRA URL download failed: {e}",
            hint=f"Check the URL {url!r} is reachable.",
        )
    logger.info(
        "Extra LoRA ready: %s (%.1f MB)",
        dest, os.path.getsize(dest) / (1024 * 1024),
    )
    return name


def _fetch_side(key: str, url: str, dest_dir: str) -> str | None:
    """One side (high or low) of a LoRA pair: URL wins over key, an
    unset side returns None."""
    if url:
        return _download_lora_from_url(url, dest_dir)
    if key:
        return _download_one_lora(key, dest_dir)
    return None


def ensure_lightx2v_loras(dest_dir: str) -> None:
    """Make sure the fixed LightX2V pair workflow.py always applies is
    present in dest_dir (ComfyUI's models/loras). The pair is bundled
    in the slim repo under loras/ and normally arrives via the Model
    Cache link step, but the cache snapshot can lag behind the repo:
    on 2026-11-26 a Wan endpoint created 17 minutes after the pair was
    uploaded still booted from a pre-upload snapshot, and every job
    failed prompt validation ('wan_lightx2v_high.safetensors' not in
    the loras list). Download any missing file from the repo itself —
    _download_lora_from_url skips files already present, and a failed
    fetch raises UploadError so the boot fails hard instead of
    generating without the acceleration the sampler settings assume."""
    for name in (LIGHTX2V_HIGH_NAME, LIGHTX2V_LOW_NAME):
        _download_lora_from_url(
            f"https://huggingface.co/{config.HF_REPO}"
            f"/resolve/main/loras/{name}",
            dest_dir,
        )


def download_extra_lora_pairs(dest_dir: str) -> list:
    """Fetch every configured extra LoRA pair (config.EXTRA_LORA_SLOTS)
    into dest_dir (ComfyUI's models/loras). Returns one entry per slot:
    a (high_name, low_name) tuple — either side may be None when only
    the other side is configured (the workflow then applies the one
    file to both branches) — or None when the slot is not configured
    at all. A configured-but-unfetchable file raises UploadError:
    silently generating without a LoRA the caller explicitly set up
    would hand back the wrong video."""
    pairs = []
    for high_key, low_key, _strength, high_url, low_url in config.EXTRA_LORA_SLOTS:
        if not (high_key or low_key or high_url or low_url):
            pairs.append(None)
            continue
        pairs.append((
            _fetch_side(high_key, high_url, dest_dir),
            _fetch_side(low_key, low_url, dest_dir),
        ))
    return pairs
