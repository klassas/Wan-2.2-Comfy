"""Input validation for the Wan 2.2 + NSFW MMAudio ComfyUI worker.

The field set is what the RunPod Media Studio app sends for video
models (see app RunPodClient.kt): flat fields prompt / negative_prompt
/ image_url / image_base64 / width / height, plus a `workflow` object
this worker accepts and ignores — it builds the graph itself in
src/workflow.py, same contract as the sibling workers.

Mode is implicit: a start image (image_base64 / image_url) makes the
job image-to-video; without one the I2V model pair runs text-to-video
(WanImageToVideo accepts a missing start_image). The audio stage is on
unless the job sets audio=false or mute=true.

Frames: Wan 2.2 runs at a fixed 16 fps and the official templates
compute length = seconds * fps + 1; the count is snapped down to the
model's 4n+1 grid here so every consumer sees the real frame count.
"""

from src import config
from src.errors import ValidationError


def _validate_image_source(inp: dict, url_key: str, b64_key: str) -> dict | None:
    """Normalize one image slot to {"url": ...} / {"base64": ...} / None."""
    has_url = bool(inp.get(url_key))
    has_b64 = bool(inp.get(b64_key))
    if has_url and has_b64:
        raise ValidationError(
            f"Provide either {url_key} or {b64_key}, not both.",
            hint="Remove one of the image source fields.",
        )
    if has_url:
        return {"url": inp[url_key]}
    if has_b64:
        return {"base64": inp[b64_key]}
    return None


def _strength(inp: dict, field: str, default: float) -> float:
    """Per-request LoRA strength (ported from the H3 worker). The LoRA
    files themselves are deploy-time choices (config.EXTRA_LORA_SLOTS);
    per request only their strengths are tunable, 0 disabling that
    slot for the job."""
    v = inp.get(field, default)
    if not isinstance(v, (int, float)) or isinstance(v, bool) \
            or v < 0 or v > 2:
        raise ValidationError(
            f"{field} must be a number between 0 and 2.",
            hint=f"You sent {field}={v!r}.",
        )
    return float(v)


def _bool_field(inp: dict, field: str, default: bool) -> bool:
    v = inp.get(field, default)
    if not isinstance(v, bool):
        raise ValidationError(f"{field} must be a boolean.")
    return v


def _text_field(inp: dict, field: str, max_len: int) -> str:
    v = inp.get(field, "")
    if not isinstance(v, str):
        raise ValidationError(f"{field} must be a string.")
    if len(v) > max_len:
        raise ValidationError(
            f"{field} exceeds max length ({max_len} chars).",
            hint="Shorten it.",
        )
    return v


def validate_input(inp: dict) -> dict:
    """Validate and normalize job input. Returns a cleaned dict with defaults applied."""
    if not isinstance(inp, dict):
        raise ValidationError(
            "Input must be a JSON object.",
            hint="Send a JSON object with at least a 'prompt' field.",
        )

    start_image = _validate_image_source(inp, "image_url", "image_base64")

    prompt = inp.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValidationError("prompt must be a string.")
    prompt = prompt.strip()
    if not prompt:
        raise ValidationError(
            "prompt is required.",
            hint="Provide a text description of the desired video.",
        )
    if len(prompt) > config.MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"prompt exceeds max length ({config.MAX_PROMPT_LENGTH} chars).",
            hint="Shorten your prompt.",
        )

    negative_prompt = _text_field(inp, "negative_prompt", config.MAX_PROMPT_LENGTH)

    seconds = inp.get("seconds", config.DEFAULT_SECONDS)
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) \
            or seconds < config.MIN_SECONDS or seconds > config.MAX_SECONDS:
        raise ValidationError(
            f"seconds must be a number between {config.MIN_SECONDS} and {config.MAX_SECONDS}.",
            hint=f"You sent seconds={seconds!r}.",
        )
    # 16 fps, 4n+1 frame grid (see module docstring).
    frames = int(round(seconds * config.FPS)) + 1
    frames = 1 + ((frames - 1) // 4) * 4

    seed = inp.get("seed", config.DEFAULT_SEED)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError("seed must be an integer.")

    steps = inp.get("steps", config.DEFAULT_STEPS)
    if not isinstance(steps, int) or isinstance(steps, bool) \
            or steps < 2 or steps > 100:
        raise ValidationError(
            "steps must be an integer between 2 and 100.",
            hint=f"You sent steps={steps!r}.",
        )

    cfg = inp.get("cfg", inp.get("cfg_scale", config.DEFAULT_CFG))
    if not isinstance(cfg, (int, float)) or isinstance(cfg, bool) \
            or cfg < 0 or cfg > 20:
        raise ValidationError(
            "cfg must be a number between 0 and 20.",
            hint=f"You sent cfg={cfg!r}.",
        )

    # Size: explicit width/height win; then the `resolution` preset;
    # then the defaults. Both-or-neither for width/height.
    has_w, has_h = "width" in inp, "height" in inp
    if has_w != has_h:
        raise ValidationError(
            "width and height must be sent together.",
            hint="Send both, or neither to use the defaults.",
        )
    if has_w:
        width, height = inp["width"], inp["height"]
    elif "resolution" in inp:
        preset = config.RESOLUTION_PRESETS.get(inp["resolution"])
        if preset is None:
            raise ValidationError(
                f"Unknown resolution preset: {inp['resolution']!r}.",
                hint=f"Known presets: {sorted(config.RESOLUTION_PRESETS)}.",
            )
        width, height = preset
    else:
        width, height = config.DEFAULT_WIDTH, config.DEFAULT_HEIGHT
    if not isinstance(width, int) or not isinstance(height, int) \
            or isinstance(width, bool) or isinstance(height, bool):
        raise ValidationError("width and height must be integers.")
    if width < config.MIN_DIMENSION or height < config.MIN_DIMENSION:
        raise ValidationError(
            f"width and height must be at least {config.MIN_DIMENSION}.",
            hint=f"You sent {width}x{height}.",
        )
    if width % config.MOD_VALUE != 0 or height % config.MOD_VALUE != 0:
        snapped_w = width // config.MOD_VALUE * config.MOD_VALUE
        snapped_h = height // config.MOD_VALUE * config.MOD_VALUE
        raise ValidationError(
            f"width and height must be divisible by {config.MOD_VALUE}.",
            hint=f"You sent {width}x{height}. Try {snapped_w}x{snapped_h}.",
        )
    if width * height > config.MAX_RESOLUTION_AREA:
        raise ValidationError(
            f"Total pixels ({width}x{height}={width * height}) exceeds max ({config.MAX_RESOLUTION_AREA}).",
            hint="Use a smaller size or increase MAX_RESOLUTION_AREA.",
        )

    # Audio stage. `audio` toggles the MMAudio branch; `mute` is the
    # app's generic no-sound flag and wins over it. Both off = the
    # graph has no audio branch at all.
    audio = _bool_field(inp, "audio", True)
    mute = _bool_field(inp, "mute", False)
    audio_prompt = _text_field(inp, "audio_prompt", config.MAX_AUDIO_PROMPT_LENGTH)
    audio_negative_prompt = _text_field(
        inp, "audio_negative_prompt", config.MAX_AUDIO_PROMPT_LENGTH
    )
    audio_steps = inp.get("audio_steps", config.DEFAULT_AUDIO_STEPS)
    if not isinstance(audio_steps, int) or isinstance(audio_steps, bool) \
            or audio_steps < 1 or audio_steps > 100:
        raise ValidationError(
            "audio_steps must be an integer between 1 and 100.",
            hint=f"You sent audio_steps={audio_steps!r}.",
        )
    audio_cfg = inp.get("audio_cfg", config.DEFAULT_AUDIO_CFG)
    if not isinstance(audio_cfg, (int, float)) or isinstance(audio_cfg, bool) \
            or audio_cfg < 0 or audio_cfg > 20:
        raise ValidationError(
            "audio_cfg must be a number between 0 and 20.",
            hint=f"You sent audio_cfg={audio_cfg!r}.",
        )

    lora_strength = _strength(inp, "lora_strength", config.EXTRA_LORA_STRENGTH)
    lora_strength_2 = _strength(inp, "lora_strength_2", config.EXTRA_LORA_STRENGTH_2)
    lora_strength_3 = _strength(inp, "lora_strength_3", config.EXTRA_LORA_STRENGTH_3)

    output_cfg = inp.get("output", {})
    if not isinstance(output_cfg, dict):
        raise ValidationError("output must be a JSON object.")

    ignored = [p for p in config.IGNORED_PARAMS if p in inp]

    return {
        "mode": "i2v" if start_image else "t2v",
        "start_image": start_image,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seconds": seconds,
        "fps": config.FPS,
        "frames": frames,
        "seed": seed,
        "steps": steps,
        "cfg_scale": float(cfg),
        "width": width,
        "height": height,
        "audio": audio and not mute,
        "mute": mute,
        "audio_prompt": audio_prompt,
        "audio_negative_prompt": audio_negative_prompt,
        "audio_steps": audio_steps,
        "audio_cfg": float(audio_cfg),
        "lora_strength": lora_strength,
        "lora_strength_2": lora_strength_2,
        "lora_strength_3": lora_strength_3,
        "ignored_params": ignored,
        "output": {
            "bucket": output_cfg.get("bucket", config.S3_BUCKET_DEFAULT),
            "prefix": output_cfg.get("prefix", "wan2.2-mmaudio/"),
            "public": output_cfg.get("public", True),
            "ttl_seconds": output_cfg.get("ttl_seconds", 0),
        },
    }
