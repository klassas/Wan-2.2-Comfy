FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Model Cache mounts the endpoint's Model repo under
# /runpod-volume/huggingface-cache/hub/ but does NOT set HF_HOME for us —
# confirmed on a live worker (both HF_HOME and HF_HUB_CACHE unset). The
# Flux/Krea/Wan2.2 workers set it here themselves; same here.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/runpod-volume/huggingface-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ComfyUI itself, via comfy-cli (its torch is replaced with a cu130 build in
# the next step).
#
# Version floor: the Wan 2.2 node set (WanImageToVideo, ModelSamplingSD3,
# KSamplerAdvanced, AudioConcat) is verified present at 0.46.2 (see
# WAN_MMAUDIO_RESEARCH.md) — the same pin the Qwen-Image / H3 Image / LTX
# workers use. Pinned to a specific release rather than "latest" so a
# future ComfyUI release cannot change node behavior under this worker
# without a deliberate version bump here. Bump only after re-verifying
# src/workflow.py's node graph against the new release's node schemas.
ARG COMFYUI_VERSION=0.46.2
ARG COMFY_DIR=/opt/ComfyUI
ENV COMFY_DIR=${COMFY_DIR}
RUN pip install --no-cache-dir comfy-cli && \
    comfy --workspace ${COMFY_DIR} --skip-prompt install \
        --version ${COMFYUI_VERSION} --cuda-version 12.8 --nvidia && \
    test -f ${COMFY_DIR}/main.py

# Swap torch for a cu130 build. On cu128 torch, ComfyUI's comfy_kitchen
# disables its CUDA and Triton kernel backends ("You need pytorch with cu130
# or higher to use optimized CUDA operations") and runs quantized weights
# through the eager fallback. torchaudio's newest cu130 build is 2.11.0, so
# the trio is pinned to the matching 2.11 set. cu130 wheels bundle their own
# CUDA runtime but need a host driver that supports CUDA 13.0 (>= 580) —
# restrict the endpoint's allowed CUDA versions accordingly.
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
# Explicit +cu130 local versions (PyPI also publishes torch==2.11.0, as a
# different CUDA build); no --force-reinstall, so ComfyUI's other pinned deps
# are only touched if torch actually requires it.
RUN python -m pip install --no-cache-dir --timeout 300 --retries 5 \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
        torch==${TORCH_VERSION}+cu130 torchaudio==${TORCHAUDIO_VERSION}+cu130 \
        torchvision==${TORCHVISION_VERSION}+cu130 && \
    python -c "import torch, torchaudio, torchvision; \
print(f'torch={torch.__version__} cuda={torch.version.cuda} torchaudio={torchaudio.__version__} torchvision={torchvision.__version__}'); \
assert torch.version.cuda and torch.version.cuda.startswith('13'), torch.version.cuda"

# Custom nodes, cloned at pinned commits (a moving main could change node
# schemas under this worker; bump deliberately, like ComfyUI itself).
#
# ComfyUI-MMAudio: phazei's fork, NOT kijai's upstream. Upstream's
# MMAudioSampler slices the first N frames of the batch for its CLIP /
# Synchformer streams, which mis-times any video whose frame rate is not
# 25 fps — Wan's 16 fps output would get audio that ends early (the
# known pitfall in WAN_MMAUDIO_RESEARCH.md). phazei's build (his PR #52
# to kijai, "Scale input frames to match CLIP and synchformer fps")
# resamples the frames uniformly over their full span from the frame
# count + duration instead. The grep below fails the build if the
# cloned tree does not contain that full-span resample.
ARG MMAUDIO_REPO=https://github.com/phazei/ComfyUI-MMAudio
ARG MMAUDIO_SHA=da86d97966e919709e815db381c05829f7993de8
ARG VHS_REPO=https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
ARG VHS_SHA=775ae691ecbab6a6d09a2c186f95d130cf8a10d3
RUN git clone ${MMAUDIO_REPO} ${COMFY_DIR}/custom_nodes/ComfyUI-MMAudio && \
    git -C ${COMFY_DIR}/custom_nodes/ComfyUI-MMAudio checkout ${MMAUDIO_SHA} && \
    git clone ${VHS_REPO} ${COMFY_DIR}/custom_nodes/ComfyUI-VideoHelperSuite && \
    git -C ${COMFY_DIR}/custom_nodes/ComfyUI-VideoHelperSuite checkout ${VHS_SHA} && \
    grep -q "uniform_indices" ${COMFY_DIR}/custom_nodes/ComfyUI-MMAudio/nodes.py && \
    grep -q '"MMAudioSampler"' ${COMFY_DIR}/custom_nodes/ComfyUI-MMAudio/nodes.py && \
    grep -q '"VHS_VideoCombine"' ${COMFY_DIR}/custom_nodes/ComfyUI-VideoHelperSuite/videohelpersuite/nodes.py && \
    grep -q '"VHS_SplitImages"' ${COMFY_DIR}/custom_nodes/ComfyUI-VideoHelperSuite/videohelpersuite/nodes.py

# Custom-node dependencies. ComfyUI-MMAudio's requirements as-is. For
# VideoHelperSuite: imageio-ffmpeg plus opencv-python-headless.
# cv2 is NOT part of ComfyUI's own dependency set (the first Wan test
# on 2026-11-26 failed with "No module named 'cv2'" and VHS never
# loaded, so VHS_VideoCombine was missing at prompt validation); the
# headless wheel provides cv2 without the system libGL that the full
# opencv-python build needs and this image does not carry.
RUN python -m pip install --no-cache-dir --timeout 300 --retries 5 \
        -r ${COMFY_DIR}/custom_nodes/ComfyUI-MMAudio/requirements.txt && \
    python -m pip install --no-cache-dir --timeout 180 --retries 5 \
        imageio-ffmpeg opencv-python-headless

# Bake the BigVGAN vocoder into the image. ComfyUI-MMAudio's
# FeatureUtilsLoader (44k mode) otherwise downloads
# nvidia/bigvgan_v2_44khz_128band_512x into models/mmaudio/nvidia/ on
# first use — mid-job. Placing it here at build time (the handler also
# pre-warms the same path at boot, covering images built without this
# step) keeps the first generation free of a surprise download.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='nvidia/bigvgan_v2_44khz_128band_512x', \
    ignore_patterns=['*3m*'], \
    local_dir='${COMFY_DIR}/models/mmaudio/nvidia/bigvgan_v2_44khz_128band_512x')" && \
    test -d ${COMFY_DIR}/models/mmaudio/nvidia/bigvgan_v2_44khz_128band_512x

# This worker's own deps (runpod, requests, websocket-client for
# src/comfy_client.py, boto3 for src/storage.py) — separate from whatever
# comfy-cli installed into the same environment for ComfyUI itself.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --timeout 180 --retries 5 \
    -r requirements.txt

# Application code
COPY src/ src/

# Verify imports and graph construction at build time. handler.py is
# excluded — importing it starts the ComfyUI subprocess and blocks on
# model linking. Everything else is pure Python with no GPU/process
# dependency, so any dependency mismatch or typo fails the build here
# instead of at the next worker cold start.
RUN python -c "\
from src.config import MODEL_TAG, HF_REPO, COMFY_DIR, model_tag; \
from src.validator import validate_input; \
from src.workflow import build_workflow, OUTPUT_NODE_ID, UNET_HIGH_NAME, MMAUDIO_MODEL_NAME; \
from src.comfy_client import ComfyClient, ComfyError; \
from src.model_link import link_model_cache, ModelLinkError; \
from src.image import fetch_image; \
from src.storage import upload_video, download_extra_lora_pairs; \
from src.errors import format_error; \
from src.security import validate_url; \
import runpod, boto3, websocket; \
wf = build_workflow(prompt='a cat', seed=1, audio=False); \
assert OUTPUT_NODE_ID in wf and 'audio' not in wf[OUTPUT_NODE_ID]['inputs'], wf[OUTPUT_NODE_ID]; \
assert 'audio_sampler' not in wf and 'load_start' not in wf, wf.keys(); \
assert wf['sampler_low']['inputs']['latent_image'] == ['sampler_high', 0], wf['sampler_low']; \
assert wf['i2v']['class_type'] == 'WanImageToVideo' and wf['i2v']['inputs']['length'] == 81, wf['i2v']; \
wf = build_workflow(prompt='x', seed=1); \
assert wf['audio_sampler']['inputs']['duration'] == 5.0, wf['audio_sampler']; \
assert wf[OUTPUT_NODE_ID]['inputs']['audio'] == ['audio_sampler', 0], wf[OUTPUT_NODE_ID]; \
wf = build_workflow(prompt='x', seed=1, frames=241, start_image_name='a.png'); \
assert wf['i2v']['inputs']['start_image'] == ['load_start', 0], wf['i2v']; \
assert wf['split_frames_1']['inputs']['split_index'] == 128, wf['split_frames_1']; \
assert wf['audio_sampler_1']['inputs']['duration'] == 8.0 and wf['audio_sampler_1']['inputs']['images'] == ['split_frames_1', 0], wf['audio_sampler_1']; \
assert wf['audio_sampler_2']['inputs']['duration'] == 7.0 and wf['audio_sampler_2']['inputs']['images'] == ['split_frames_1', 2], wf['audio_sampler_2']; \
assert wf[OUTPUT_NODE_ID]['inputs']['audio'] == ['audio_concat_1', 0], wf[OUTPUT_NODE_ID]; \
wf = build_workflow(prompt='x', seed=1, extra_loras=[(('h.safetensors', 'l.safetensors'), 0.5), (('only.safetensors', None), 1.0), None]); \
assert wf['sampling_high']['inputs']['model'] == ['lora_high_2', 0], wf['sampling_high']; \
assert wf['sampling_low']['inputs']['model'] == ['lora_low_2', 0], wf['sampling_low']; \
assert wf['lora_low']['inputs']['lora_name'] == 'l.safetensors', wf['lora_low']; \
assert wf['lora_low_2']['inputs']['lora_name'] == 'only.safetensors', wf['lora_low_2']; \
p = validate_input({'prompt': 'x', 'workflow': {'1': {}}}); \
assert p['mode'] == 't2v' and p['frames'] == 81 and p['audio'] is True and p['ignored_params'] == ['workflow'], p; \
p = validate_input({'prompt': 'x', 'image_base64': 'AAAA', 'seconds': 15, 'mute': True}); \
assert p['mode'] == 'i2v' and p['frames'] == 241 and p['audio'] is False, p; \
p = validate_input({'prompt': 'x', 'resolution': '480p'}); \
assert (p['width'], p['height']) == (960, 544), p; \
print(f'All imports OK (tag={MODEL_TAG} model={model_tag()} repo={HF_REPO} comfy_dir={COMFY_DIR})')"

# Only relevant when SERVE_MODE=api (the on-demand Pod deploy path, see
# src/watchdog.py). Serverless (the default) never
# binds this — EXPOSE is documentation for `docker run -p`/Pod port
# mapping, not a switch.
EXPOSE 8000

CMD ["python", "-u", "src/handler.py"]
