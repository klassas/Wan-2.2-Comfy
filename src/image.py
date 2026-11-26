"""Input-image download and decode.

A job carries at most one input image: the start frame (the
validator's normalized {"url": ...} / {"base64": ...} source). No
resizing here: WanImageToVideo rescales the start image to the
latent's pixel size itself, so the image is uploaded to ComfyUI
as-is.
"""

import base64
import io

import requests
from PIL import Image

from src import config
from src.errors import DownloadError, ValidationError
from src.security import validate_url


def fetch_image_source(source: dict | None) -> Image.Image | None:
    """Fetch one normalized source ({"url"} / {"base64"}). None in, None out."""
    if not source:
        return None
    if source.get("url"):
        return _download_image(source["url"])
    if source.get("base64"):
        return _decode_base64(source["base64"])
    return None


def fetch_image(validated_input: dict) -> Image.Image | None:
    """Fetch the start frame. None when the job provided none (T2V)."""
    return fetch_image_source(validated_input.get("start_image"))


def _download_image(url: str) -> Image.Image:
    validate_url(url)

    try:
        resp = requests.get(
            url,
            timeout=config.IMAGE_DOWNLOAD_TIMEOUT,
            stream=True,
            headers={"User-Agent": "WanMMAudio-Worker/1.0"},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise DownloadError(
            f"Failed to download image: {e}",
            hint="Check that the URL is accessible and returns an image.",
        )

    content_type = resp.headers.get("Content-Type", "")
    if content_type and not content_type.startswith("image/"):
        raise DownloadError(
            f"URL returned non-image Content-Type: {content_type}",
            hint="Provide a URL that returns an image (png/jpg/webp).",
        )

    chunks = []
    total = 0
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        total += len(chunk)
        if total > config.MAX_IMAGE_BYTES:
            raise DownloadError(
                f"Image exceeds max size ({config.MAX_IMAGE_BYTES // (1024*1024)} MB).",
                hint="Use a smaller image.",
            )
        chunks.append(chunk)

    return _open_image(b"".join(chunks))


def _decode_base64(b64_str: str) -> Image.Image:
    # Accept data-URL prefix as well as raw base64.
    if b64_str.startswith("data:"):
        _, _, b64_str = b64_str.partition(",")

    try:
        data = base64.b64decode(b64_str)
    except Exception:
        raise ValidationError(
            "Invalid base64 encoding for image_base64.",
            hint="Send raw base64 or a data: URL.",
        )

    if len(data) > config.MAX_IMAGE_BYTES:
        raise ValidationError(
            f"Decoded image exceeds max size ({config.MAX_IMAGE_BYTES // (1024*1024)} MB).",
            hint="Use a smaller image.",
        )

    return _open_image(data)


def _open_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise DownloadError(
            "Cannot decode image data.",
            hint="Ensure the image is a valid PNG, JPEG, or WebP file.",
        )

    if img.format and img.format.upper() not in ("PNG", "JPEG", "JPG", "WEBP"):
        raise ValidationError(
            f"Unsupported image format: {img.format}",
            hint="Supported formats: PNG, JPEG, WebP.",
        )

    w, h = img.size
    if w * h > config.MAX_IMAGE_PIXELS:
        raise ValidationError(
            f"Image too large: {w}x{h} ({w*h} pixels).",
            hint=f"Max pixels: {config.MAX_IMAGE_PIXELS}.",
        )

    return img.convert("RGB")
