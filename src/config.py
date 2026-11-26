"""Configuration and defaults for the Wan 2.2 + NSFW MMAudio worker.

Mirrors the sibling workers' config conventions (env names, S3 block,
Pod on-demand mode) wherever the workers share a concept, so the
endpoint setup steps and the app's expectations carry over unchanged.
Wan/MMAudio-specific fields (model filenames, sampler defaults) live
in src/workflow.py, same split as the sibling workers.
"""

import os


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# -- Model / ComfyUI process ──────────────────────────────────────────────────
# Lowercase deliberately — see src/model_link.py's REPO_ID comment. Must match
# the literal string configured in the RunPod endpoint's Model field, not the
# Hub's mixed-case canonical id. The slim repo holds only the nine needed
# files (Wan I2V pair + umt5 + wan_2.1_vae + the four MMAudio files,
# ~40.7 GB); the I2V pair doubles as T2V (WanImageToVideo runs without a
# start image), so no T2V pair is carried. Wan 2.2 is Apache-2.0 with no
# territory restrictions and the repo is public, so no HF_TOKEN is needed.
HF_REPO = os.getenv("HF_REPO", "klassas/wan-2.2-mmaudio-slim")
# Where the Dockerfile installs ComfyUI itself (core only, no models baked
# in — weights come from Model Cache via src/model_link.py at runtime).
COMFY_DIR = os.getenv("COMFY_DIR", "/opt/ComfyUI")
COMFY_HOST = os.getenv("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.getenv("COMFY_PORT", "8188"))
# Cold start: ComfyUI's own process boot (before any model weights load —
# those load lazily per-prompt) plus this worker's model_link.py symlinking.
COMFY_BOOT_TIMEOUT_S = int(os.getenv("COMFY_BOOT_TIMEOUT_S", "120"))
# Per-job ceiling: the first prompt after boot pays weight-load time
# (~36 GB of fp8 weights) on top of two sampling passes plus the audio
# stage, so this matches the endpoint's 1800 s execution timeout.
COMFY_INFERENCE_TIMEOUT_S = int(os.getenv("COMFY_INFERENCE_TIMEOUT_S", "1800"))
# A known ComfyUI regression around pinned-memory allocation has been
# reported fixed by --disable-pinned-memory. Off by default; flip via env
# var without a rebuild if cold starts show it.
COMFY_DISABLE_PINNED_MEMORY = _env_bool("COMFY_DISABLE_PINNED_MEMORY", "0")

MODEL_TAG = os.getenv("MODEL_TAG", "wan-2.2-mmaudio-comfy")


def model_tag() -> str:
    return MODEL_TAG


# -- Inference defaults ───────────────────────────────────────────────────────
# Wan 2.2 runs at a fixed 16 fps; frames = seconds * 16 + 1 (the official
# templates' ComfyMathExpression "floor(seconds * fps + 1)").
FPS = 16
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "960"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "544"))
DEFAULT_SECONDS = 5
DEFAULT_SEED = -1
# LightX2V 4-step sampler settings: the slim set bundles the LightX2V
# I2V LoRA pair (loras/wan_lightx2v_high|low.safetensors, applied
# unconditionally in workflow.py), so sampling runs 8 steps total split
# evenly between the high-noise and low-noise models at cfg 1.0,
# ModelSamplingSD3 shift 5.0. (The first Wan test on 2026-11-26 used
# the templates' standard 20 steps / cfg 3.5 and took ~68 s per step
# on a 48 GB Ampere GPU — about 25 minutes per clip. That is why the
# 4-step mode is now the default.)
DEFAULT_STEPS = 8
DEFAULT_CFG = 1.0
SAMPLING_SHIFT = 5.0
# MMAudio stage defaults (the sampler node's own defaults).
DEFAULT_AUDIO_STEPS = 25
DEFAULT_AUDIO_CFG = 4.5
# MMAudio was trained on ~8 s clips; longer audio is generated in
# windows of at most this many seconds and concatenated (see
# src/workflow.py). At 16 fps one window spans 128 frames.
AUDIO_WINDOW_SECONDS = 8.0

# Output sizes must be multiples of this, with the same area cap the
# H3 worker uses (research contract: 32-grid, 1,032,192 px max).
MOD_VALUE = 32
MAX_RESOLUTION_AREA = int(os.getenv("MAX_RESOLUTION_AREA", str(1032192)))
MIN_DIMENSION = 128

# Resolution presets, same table as the MiniMax-H3 worker: the app may
# send `resolution` instead of explicit width/height.
RESOLUTION_PRESETS = {
    "768p": (1344, 768),
    "768p-portrait": (768, 1344),
    "720p": (1344, 768),
    "720p-portrait": (768, 1344),
    "square": (992, 992),
    "480p": (960, 544),
    "480p-portrait": (544, 960),
}

# -- Limits ───────────────────────────────────────────────────────────────────
# The app's durations are 5 / 8 / 10 / 15 s plus a Wan-only 16 s
# (two full 8 s MMAudio windows); the ceiling is a guard rail
# (MAX_SECONDS can only lower it).
SECONDS_CEILING = 16
MAX_SECONDS = min(int(os.getenv("MAX_SECONDS", "16")), SECONDS_CEILING)
MIN_SECONDS = 1
# The app composes long tile-based prompts (character/scene blocks), so
# the prompt ceiling is much higher than the H3 worker's 2000.
MAX_PROMPT_LENGTH = 8000
MAX_AUDIO_PROMPT_LENGTH = 2000
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 8192 * 8192
IMAGE_DOWNLOAD_TIMEOUT = 10
UPLOAD_TIMEOUT = 60

# Fields the app sends that this worker accepts and drops: `workflow` is
# the placeholder asset the app still ships for old third-party
# endpoints — this worker builds its graph in code (src/workflow.py).
IGNORED_PARAMS = ("workflow",)

# -- S3 / object storage ──────────────────────────────────────────────────────
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_REGION = os.getenv("S3_REGION", "auto")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")
S3_BUCKET_DEFAULT = os.getenv("S3_BUCKET_DEFAULT", "")
S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL", "")

# -- On-demand Pod mode (SERVE_MODE=api, see src/watchdog.py) ─────────────────
SERVE_MODE = os.getenv("SERVE_MODE", "serverless")
POD_API_PORT = os.getenv("POD_API_PORT", "8000")
POD_IDLE_TIMEOUT_S = int(os.getenv("POD_IDLE_TIMEOUT_S", "180"))
POD_MAX_LIFETIME_S = int(os.getenv("POD_MAX_LIFETIME_S", "10800"))
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")


# -- Extra (user) LoRAs --------------------------------------------------------
# Up to three LoRA slots, ported from the MiniMax-H3 / LTX workers (files
# fetched at boot by storage.download_extra_lora_pairs, so swapping LoRAs
# is an env change + worker restart, never a rebuild). The Wan 2.2
# difference: LoRAs ship as HIGH-noise / LOW-noise file pairs, one file
# per sampling branch. Per slot:
#
#   EXTRA_LORA_KEY{,_2,_3}       high-noise file's object key in the bucket
#   EXTRA_LORA_KEY{,_2,_3}_LOW   low-noise file's object key
#   EXTRA_LORA_URL{,_2,_3}       high-noise file's direct URL (wins over key)
#   EXTRA_LORA_URL{,_2,_3}_LOW   low-noise file's direct URL
#
# If only one side of a pair is configured, that file is applied to BOTH
# branches (single-file LoRAs exist and are used that way). Per-request
# strengths come from the job (lora_strength{,_2,_3}, 0 disables the slot
# for that job); EXTRA_LORA_STRENGTH{,_2,_3} are the defaults.
def _lora_slot_env(suffix: str) -> tuple:
    return (
        os.getenv(f"EXTRA_LORA_KEY{suffix}", ""),
        os.getenv(f"EXTRA_LORA_KEY{suffix}_LOW", ""),
        float(os.getenv(f"EXTRA_LORA_STRENGTH{suffix}", "0.5")),
        os.getenv(f"EXTRA_LORA_URL{suffix}", ""),
        os.getenv(f"EXTRA_LORA_URL{suffix}_LOW", ""),
    )


# (high_key, low_key, default_strength, high_url, low_url) per slot.
EXTRA_LORA_SLOTS = [_lora_slot_env(""), _lora_slot_env("_2"), _lora_slot_env("_3")]
EXTRA_LORA_STRENGTH = EXTRA_LORA_SLOTS[0][2]
EXTRA_LORA_STRENGTH_2 = EXTRA_LORA_SLOTS[1][2]
EXTRA_LORA_STRENGTH_3 = EXTRA_LORA_SLOTS[2][2]
