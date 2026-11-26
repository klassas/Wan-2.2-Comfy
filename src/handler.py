"""RunPod Serverless handler for the Wan 2.2 + NSFW MMAudio worker.

Same architecture as the sibling workers: this process loads no model
weights itself. It launches ComfyUI as a subprocess and talks to it
over HTTP/WebSocket (src/comfy_client.py); ComfyUI resolves and loads
weights lazily, per node, the first time a prompt graph references
them. What this handler owns is the RunPod job contract: validating
input (src/validator.py), fetching the optional start frame
(src/image.py), building the node graph (src/workflow.py — Wan video
stage + MMAudio audio stage + VHS mux in ONE graph), running it, and
uploading the resulting MP4 (src/storage.py).

Mode is implicit: a start frame makes the job image-to-video; without
one the I2V model pair runs text-to-video. The audio stage runs
unless the job sets audio=false or mute=true.
"""

import io
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import runpod

from src import config, watchdog
from src.comfy_client import ComfyClient, ComfyError, new_client_id
from src.errors import InferenceError, ValidationError, format_error
from src.image import fetch_image
from src.model_link import ModelLinkError, link_model_cache
from src.storage import download_extra_lora_pairs, upload_video
from src.validator import validate_input
from src.workflow import (
    CLIP_NAME,
    MMAUDIO_CLIP_NAME,
    MMAUDIO_MODEL_NAME,
    MMAUDIO_SYNCHFORMER_NAME,
    MMAUDIO_VAE_NAME,
    OUTPUT_NODE_ID,
    UNET_HIGH_NAME,
    UNET_LOW_NAME,
    VAE_NAME,
    build_workflow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s",
)
logger = logging.getLogger("wan-2.2-mmaudio-comfy")

_COMFY_READY = threading.Event()
_COMFY_ERROR: Exception | None = None
_comfy_process: subprocess.Popen | None = None
_client = ComfyClient(host=config.COMFY_HOST, port=config.COMFY_PORT)
_EXTRA_LORA_PAIRS: list = [None, None, None]

# The nine slim-set files the graph loads, by ComfyUI models/ subdir.
_MODEL_FILES = (
    ("diffusion_models", UNET_HIGH_NAME),
    ("diffusion_models", UNET_LOW_NAME),
    ("text_encoders", CLIP_NAME),
    ("vae", VAE_NAME),
    ("mmaudio", MMAUDIO_MODEL_NAME),
    ("mmaudio", MMAUDIO_VAE_NAME),
    ("mmaudio", MMAUDIO_SYNCHFORMER_NAME),
    ("mmaudio", MMAUDIO_CLIP_NAME),
)

# ComfyUI-MMAudio's FeatureUtilsLoader (44k mode) auto-downloads the
# BigVGAN vocoder into models/mmaudio/nvidia/<repo name> on first use.
# Doing that mid-job would stall (or fail) the first generation, so the
# boot pre-warms the exact same directory (the Dockerfile also bakes a
# copy into the image; this covers images built without it).
BIGVGAN_REPO_ID = "nvidia/bigvgan_v2_44khz_128band_512x"
BIGVGAN_DIR_NAME = os.path.join("nvidia", "bigvgan_v2_44khz_128band_512x")


def _prewarm_bigvgan() -> None:
    """Download the BigVGAN vocoder next to the MMAudio models if it is
    not already there. Best-effort: audio=false jobs never need it, and
    the loader would retry the download itself on the first audio job —
    a failure here is logged loudly, not fatal."""
    dest = os.path.join(config.COMFY_DIR, "models", "mmaudio", BIGVGAN_DIR_NAME)
    if os.path.exists(dest):
        logger.info("BigVGAN vocoder already present: %s", dest)
        return
    try:
        from huggingface_hub import snapshot_download

        logger.info("Pre-warming BigVGAN vocoder download -> %s", dest)
        snapshot_download(
            repo_id=BIGVGAN_REPO_ID,
            ignore_patterns=["*3m*"],
            local_dir=dest,
        )
        logger.info("BigVGAN vocoder ready: %s", dest)
    except Exception:
        logger.exception(
            "BigVGAN pre-warm failed; the first audio job will retry "
            "the download inside ComfyUI-MMAudio's loader"
        )


# -- ComfyUI subprocess boot ───────────────────────────────────────────────────

_CUDNN_OPTOUT_NODE = '''"""Worker-installed: keep SDPA away from the cuDNN backend.

On this stack (torch cu130 + cuDNN 9) the cuDNN SDPA backend fails inside
attention with "cuDNN Frontend error: No valid execution plans built"
(seen on RTX A5000 and PRO 6000 MIG with Qwen-Image 2.1). Setting
torch.backends.cuda.enable_cudnn_sdp(False) alone is NOT enough:
ComfyUI 0.46's comfy/ops.py wraps every SDPA call in
sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True), and that priority
list re-enables the cuDNN backend inside the context. So this module
also removes CUDNN_ATTENTION from comfy.ops.SDPA_BACKEND_PRIORITY,
leaving FLASH -> EFFICIENT -> MATH. Convolutions keep using cuDNN.
Written by the worker's handler.py at boot; do not edit here.
"""

try:
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass

try:
    import comfy.ops
    from torch.nn.attention import SDPBackend

    comfy.ops.SDPA_BACKEND_PRIORITY = [
        backend
        for backend in comfy.ops.SDPA_BACKEND_PRIORITY
        if backend != SDPBackend.CUDNN_ATTENTION
    ]
except Exception:
    pass

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
'''


def _install_cudnn_optout_node() -> None:
    """Drop the cuDNN SDPA opt-out into ComfyUI's custom_nodes/ so ComfyUI
    imports it during its own startup (after torch is loaded, before any
    generation). Best-effort: a failure here must not block the boot."""
    try:
        node_dir = os.path.join(
            config.COMFY_DIR, "custom_nodes", "zz_cudnn_sdp_optout"
        )
        os.makedirs(node_dir, exist_ok=True)
        with open(os.path.join(node_dir, "__init__.py"), "w") as f:
            f.write(_CUDNN_OPTOUT_NODE)
        logger.info("Installed cuDNN SDPA opt-out custom node in %s", node_dir)
    except Exception:
        logger.exception("Could not install cuDNN SDPA opt-out node; continuing")


def _boot_comfy_bg() -> None:
    """Symlink the Model Cache snapshot into ComfyUI's models/ tree, fetch
    any env-configured LoRA pairs, pre-warm the BigVGAN vocoder, launch
    ComfyUI, and wait for its HTTP server to answer. Runs on a
    background thread so request validation can overlap the boot, same
    pattern as the sibling workers."""
    global _COMFY_ERROR, _comfy_process, _EXTRA_LORA_PAIRS
    try:
        logger.info("Linking Model Cache snapshot into ComfyUI models/...")
        snapshot_root = link_model_cache(os.path.join(config.COMFY_DIR, "models"))
        logger.info("Model weights linked from %s", snapshot_root)

        # Fail-hard by design (storage.download_extra_lora_pairs): a
        # LoRA the deployer configured but the worker cannot fetch must
        # stop the boot, not silently generate without it.
        _EXTRA_LORA_PAIRS = download_extra_lora_pairs(
            os.path.join(config.COMFY_DIR, "models", "loras")
        )
        for _pair, (_hk, _lk, _default, _hu, _lu) in zip(
            _EXTRA_LORA_PAIRS, config.EXTRA_LORA_SLOTS
        ):
            if _pair:
                logger.info(
                    "Extra LoRA pair enabled: high=%s low=%s (default strength %.2f)",
                    _pair[0], _pair[1], _default,
                )

        _prewarm_bigvgan()

        logger.info("Starting ComfyUI subprocess in %s", config.COMFY_DIR)
        comfy_args = [
            sys.executable, "main.py",
            "--listen", config.COMFY_HOST,
            "--port", str(config.COMFY_PORT),
        ]
        if config.COMFY_DISABLE_PINNED_MEMORY:
            comfy_args.append("--disable-pinned-memory")
        # The cuDNN SDPA opt-out must NOT be applied by importing torch
        # before ComfyUI starts (the old sitecustomize.py route): ComfyUI
        # 0.46's cuda_malloc module sets
        # PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync before torch is
        # imported, and PyTorch asserts at CUDA init when the allocator
        # backend parsed at load time (native, because torch was already
        # imported) differs from the runtime one (cudaMallocAsync) —
        # ComfyUI dies instantly at startup. A custom node is imported by
        # ComfyUI itself during startup, after torch, so the flag lands
        # before any sampling call instead.
        _install_cudnn_optout_node()
        _comfy_process = subprocess.Popen(
            comfy_args,
            cwd=config.COMFY_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(target=_pipe_comfy_logs, daemon=True).start()

        _client.wait_until_alive(config.COMFY_BOOT_TIMEOUT_S)
        logger.info("ComfyUI is up.")
    except (ModelLinkError, ComfyError) as e:
        _COMFY_ERROR = e
        logger.exception("ComfyUI boot failed")
    except BaseException as e:
        _COMFY_ERROR = e
        logger.exception("ComfyUI boot failed (unexpected)")
    finally:
        _COMFY_READY.set()


def _pipe_comfy_logs() -> None:
    """Forward the ComfyUI subprocess's own stdout/stderr into this worker's
    log stream — otherwise its cold-start progress and any node errors are
    invisible in RunPod's worker log."""
    if _comfy_process is None or _comfy_process.stdout is None:
        return
    for line in _comfy_process.stdout:
        logger.info("[comfyui] %s", line.rstrip())


def _wait_for_comfy() -> None:
    _COMFY_READY.wait()
    if _COMFY_ERROR is not None:
        raise InferenceError(f"ComfyUI failed to start: {_COMFY_ERROR}")
    if _comfy_process is not None and _comfy_process.poll() is not None:
        raise InferenceError(
            f"ComfyUI process exited early (code {_comfy_process.returncode})."
        )


# -- Single job ────────────────────────────────────────────────────────────────

def _run_job(job_id: str, params: dict) -> dict:
    timings = {}

    # Fail before any GPU work if the Model Cache snapshot does not carry
    # the nine files this graph loads: ComfyUI would only reject the
    # graph after the client has waited on the whole queue round-trip.
    models_dir = os.path.join(config.COMFY_DIR, "models")
    for subdir, filename in _MODEL_FILES:
        if not os.path.exists(os.path.join(models_dir, subdir, filename)):
            raise ValidationError(
                f"Model file {filename} is not available on this worker.",
                hint=(
                    "The endpoint's Model Cache snapshot lacks it. Check the "
                    "endpoint's Model field points at a repo containing "
                    f"{subdir}/{filename} (see worker-wan/README.md)."
                ),
            )

    t0 = time.perf_counter()
    start_name = None
    start_img = fetch_image(params)
    if start_img is not None:
        buf = io.BytesIO()
        start_img.save(buf, format="PNG")
        start_name = _client.upload_image(buf.getvalue(), f"{job_id}_start.png")
    timings["preprocess_s"] = round(time.perf_counter() - t0, 3)

    seed = params["seed"]
    if seed < 0:
        import random
        seed = random.randint(0, 2**32 - 1)

    workflow = build_workflow(
        prompt=params["prompt"],
        negative_prompt=params["negative_prompt"],
        width=params["width"],
        height=params["height"],
        frames=params["frames"],
        seed=seed,
        steps=params["steps"],
        cfg=params["cfg_scale"],
        shift=config.SAMPLING_SHIFT,
        start_image_name=start_name,
        extra_loras=[
            (
                _EXTRA_LORA_PAIRS[i]
                if params[f"lora_strength{'_' + str(i + 1) if i else ''}"] > 0 else None,
                params[f"lora_strength{'_' + str(i + 1) if i else ''}"],
            )
            for i in range(3)
        ],
        audio=params["audio"],
        audio_prompt=params["audio_prompt"],
        audio_negative_prompt=params["audio_negative_prompt"],
        audio_steps=params["audio_steps"],
        audio_cfg=params["audio_cfg"],
        filename_prefix=f"wan22/{job_id}",
    )

    client_id = new_client_id()
    t0 = time.perf_counter()
    try:
        prompt_id = _client.queue(workflow, client_id)
        _client.wait_for_completion(prompt_id, client_id, config.COMFY_INFERENCE_TIMEOUT_S)
        video_bytes = _client.fetch_output(prompt_id, OUTPUT_NODE_ID)
    except ComfyError as e:
        raise InferenceError(str(e))
    timings["inference_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        mp4_path = f.name
    try:
        out_cfg = params["output"]
        video_url = upload_video(
            local_path=mp4_path,
            job_id=job_id,
            bucket=out_cfg["bucket"],
            prefix=out_cfg["prefix"],
            public=out_cfg["public"],
            ttl_seconds=out_cfg["ttl_seconds"],
        )
    finally:
        try:
            os.remove(mp4_path)
        except OSError:
            pass
    timings["upload_s"] = round(time.perf_counter() - t0, 3)
    timings["total_s"] = round(sum(timings.values()), 3)

    meta = {
        "mode": params["mode"],
        "audio": params["audio"],
        "width": params["width"],
        "height": params["height"],
        "seconds": params["seconds"],
        "fps": params["fps"],
        "frames": params["frames"],
        "seed": seed,
        "steps": params["steps"],
        "loras": [
            {
                "high": _EXTRA_LORA_PAIRS[i][0] if _EXTRA_LORA_PAIRS[i] else None,
                "low": _EXTRA_LORA_PAIRS[i][1] if _EXTRA_LORA_PAIRS[i] else None,
                "strength": params[f"lora_strength{'_' + str(i + 1) if i else ''}"] if _EXTRA_LORA_PAIRS[i] else 0,
            }
            for i in range(3)
        ],
        "model": config.model_tag(),
        "ignored_params": params["ignored_params"],
    }

    return {
        "status": "completed",
        "video_url": video_url,
        "meta": meta,
        "timing": timings,
    }


# -- Entry point ──────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """RunPod serverless handler.

    Text-to-video: {"input": {"prompt": "..."}}
    Image-to-video: {"input": {"prompt": "...", "image_base64": "..."}}
    Silent:        {"input": {"prompt": "...", "audio": false}}
    Health:        {"input": {"op": "health"}}  (SERVE_MODE=api / Pod deploy path)
    """
    job_id = job.get("id", "unknown")
    inp = job.get("input", {})

    watchdog.touch()

    if inp.get("op") == "health":
        if _COMFY_ERROR is not None:
            return {"status": "error", "error": str(_COMFY_ERROR)}
        if not _COMFY_READY.is_set():
            return {"status": "loading"}
        return {"status": "ok"}

    logger.info("Job %s started", job_id)

    try:
        _wait_for_comfy()
        params = validate_input(inp)
        result = _run_job(job_id, params)
        logger.info("Job %s completed", job_id)
        return result
    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        return format_error(exc)


# -- Startup ───────────────────────────────────────────────────────────────────

def _probe(msg: str) -> None:
    sys.stderr.write(f"[startup] {msg}\n")
    sys.stderr.flush()


try:
    _probe("entrypoint reached")
    threading.Thread(target=_boot_comfy_bg, name="comfy-boot", daemon=True).start()

    # No-op unless SERVE_MODE=api and RUNPOD_POD_ID/RUNPOD_API_KEY are present.
    watchdog.start()

    if config.SERVE_MODE == "api":
        sys.argv += [
            "--rp_serve_api",
            "--rp_api_host", "0.0.0.0",
            "--rp_api_port", config.POD_API_PORT,
        ]
        _probe(f"SERVE_MODE=api: rp_serve_api on port {config.POD_API_PORT}")
except BaseException as _e:
    sys.stderr.write(f"[startup] FATAL during warm-up: {type(_e).__name__}: {_e}\n")
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    raise

# KEEP THIS CALL AT MODULE LEVEL, UNINDENTED — RunPod's GitHub integration
# statically scans the repo for it and refuses to deploy ("handler not
# found") when it is wrapped in a try/except or a function.
runpod.serverless.start({"handler": handler})
