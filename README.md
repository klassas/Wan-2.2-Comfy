# Wan 2.2 + NSFW MMAudio Worker (worker-wan)

RunPod Serverless worker for **Wan 2.2 video generation with synced
sound** in the RunPod Media Studio app, built in the same fkky pattern
as the sibling workers (`worker-qwen/`, `worker-h3-image/`,
`worker-ltx/`): a GitHub-deployed image that installs ComfyUI, links
the endpoint's **Model Cache** snapshot into ComfyUI's `models/` tree
at boot (no network volume, no monthly fee), builds the ComfyUI graph
in code from simple fields, and uploads the result MP4 to the S3
bucket, returning a presigned URL.

One job does both stages in **one ComfyUI process, one graph** (no
intermediate file round-trip): Wan 2.2 generates the silent video,
NSFW MMAudio generates sound that follows the picture, and
VHS_VideoCombine muxes frames + audio into the returned MP4.

```
worker-wan/
├── Dockerfile          ComfyUI 0.46.2 + torch cu130 + ComfyUI-MMAudio
│                       (phazei's fps-aware build) + VideoHelperSuite,
│                       BigVGAN vocoder baked in, no model weights
├── requirements.txt    runpod / requests / websocket-client / boto3
└── src/
    ├── handler.py      RunPod entry point; boots ComfyUI as a subprocess,
    │                   pre-warms the BigVGAN vocoder download
    ├── config.py       Env-driven config (HF_REPO, S3_*, LoRA pair slots)
    ├── validator.py    App-shaped input -> cleaned params
    ├── workflow.py     Wan two-branch graph + MMAudio stage, built per job
    ├── comfy_client.py ComfyUI HTTP/WebSocket client
    ├── model_link.py   Model Cache snapshot -> ComfyUI models/ symlinks
    ├── image.py        Start-frame download + decode
    ├── storage.py      S3 upload + presigned URL, LoRA-pair download
    ├── security.py     SSRF guard for image URLs
    ├── errors.py       Shared error contract with the sibling workers
    └── watchdog.py     Pod-mode self-termination (no-op on Serverless)
```

## Model set (slim repo)

Nine files (~40.7 GB total) in the public slim repo
**`klassas/wan-2.2-mmaudio-slim`**, in ComfyUI folder layout. Only the
I2V diffusion pair is carried: it doubles as T2V (WanImageToVideo runs
without a start image), which halves the repo versus carrying the T2V
pair too — the trade-off recorded in WAN_MMAUDIO_RESEARCH.md.

| File | Role |
|---|---|
| `diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | Wan 2.2 14B, first-half (high-noise) model |
| `diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | Wan 2.2 14B, second-half (low-noise) model |
| `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | text encoder |
| `vae/wan_2.1_vae.safetensors` | video VAE (2.2 uses the 2.1 VAE) |
| `mmaudio/mmaudio_large_44k_nsfw_gold_8.5k_final_fp16.safetensors` | NSFW MMAudio main checkpoint (from `phazei/NSFW_MMaudio`) |
| `mmaudio/mmaudio_vae_44k_fp16.safetensors` | MMAudio audio VAE (from `Kijai/MMAudio_safetensors`) |
| `mmaudio/mmaudio_synchformer_fp16.safetensors` | MMAudio sync model |
| `mmaudio/apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors` | MMAudio CLIP model |

(The ninth file is the BigVGAN vocoder — not in the slim repo; the
Dockerfile bakes `nvidia/bigvgan_v2_44khz_128band_512x` into the image
under `models/mmaudio/nvidia/`, and the handler re-downloads it at
boot if it is ever missing.)

Licenses: Wan 2.2 is Apache-2.0 with **no territory restrictions** —
this endpoint needs **no data-center restriction**. MMAudio's weights
are CC-BY-NC at the root, so this worker is for personal use only.

## Endpoint settings

Create it the same way as the Qwen / H3 Image / LTX endpoints
(Serverless → "Deploy from a GitHub repository" → this repo, branch
`main`, Dockerfile path `/Dockerfile`, Queue mode):

- **GPU: 48 GB tier** (all types in the tier checked), GPU count 1,
  Max workers 1, Active workers 0
- **Cached model:** `klassas/wan-2.2-mmaudio-slim`
- **Container disk:** 20 GB (weights live in Model Cache, not the disk)
- **Idle timeout:** 60 s · **Initialization timeout:** 420 s
- **Execution timeout:** enabled, **1800** s
- **FlashBoot:** OFF (the free allowance is already used by the other
  endpoints; never accept a paid-tier dialog)
- **Data centers:** no restriction needed (see license note above)
- **Environment variables** (Edit endpoint → Raw editor, after
  creation — the create form does not reliably persist them):

```
S3_ENDPOINT=...
S3_REGION=...
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_BUCKET_DEFAULT=...
HF_REPO=klassas/wan-2.2-mmaudio-slim
```

No `HF_TOKEN` line is needed: the slim repo is public and the models
load from the local Model Cache. Optional LoRA pair slots: see below.

After the endpoint is created, **wait until its build shows Completed
and the endpoint shows Ready** before testing (a test sent while it
is still Initializing just queues and wastes time). Then enter the
endpoint's ID in the app's Settings tab and run one short test
generation (5 s, no image, audio on) with the app kept open.

## Input contract

Flat fields; the graph is built in code by `src/workflow.py`.

| Field | Meaning |
|---|---|
| `prompt` (required) | Positive prompt for the video |
| `negative_prompt` | Negative prompt (Wan actually uses it, unlike H3) |
| `seconds` | 1–16, default 5 (the app sends 5 / 8 / 10 / 15, plus a Wan-only 16 = two full 8 s audio windows). Frames = seconds × 16 fps + 1, snapped to the 4n+1 grid |
| `width` / `height` | Both or neither; multiples of 32, area ≤ 1,032,192. Default 960×544 |
| `resolution` | Preset instead of width/height, same table as the H3 worker (`480p` 960×544, `480p-portrait`, `720p`/`768p` 1344×768, `720p-portrait`/`768p-portrait`, `square` 992×992). Explicit width/height win |
| `image_base64` / `image_url` | Start frame → image-to-video. Absent → text-to-video on the I2V pair |
| `seed` | Video seed (-1 = random). The audio stage reuses it (window 2 uses seed+1) |
| `steps` | Video steps, default 20, split evenly between the high/low models (the official templates' standard mode) |
| `cfg` (or `cfg_scale`) | Video CFG, default 3.5 |
| `lora_strength` / `lora_strength_2` / `lora_strength_3` | 0–2 per-request strengths for the env-configured LoRA pair slots; 0 disables that slot for the job |
| `audio` | Default true. false = skip the MMAudio stage, return a silent MP4 |
| `audio_prompt` | Sound description for MMAudio (e.g. "wet sounds, heavy breathing") |
| `audio_negative_prompt` | Sounds to avoid (e.g. "music, speech") |
| `audio_steps` / `audio_cfg` | MMAudio sampler settings, defaults 25 / 4.5 |
| `mute` | true = silent MP4 (wins over `audio`) |
| `output` | `{"bucket", "prefix", "public": false, "ttl_seconds"}` — same B2 presigned-URL block as the other workers (default prefix `wan2.2-mmaudio/`) |

Response: `{"status": "completed", "video_url": "<presigned B2 URL>",
"meta": {...}, "timing": {...}}` — the same `video_url` key the H3 /
LTX workers return, so the app's output handling is unchanged. (The
single-graph design reports one combined `inference_s`; the video and
audio stages cannot be timed separately inside one ComfyUI prompt.)

## LoRA (high/low pairs)

Wan 2.2 LoRAs ship as **pairs**: one file for the high-noise model,
one for the low-noise model. Each of the three slots is configured at
deploy time with env vars (a set URL wins over the slot's key, per
side):

- `EXTRA_LORA_KEY` / `EXTRA_LORA_KEY_LOW` — bucket object keys for the
  high / low files (slots 2 and 3 add `_2` / `_3` before `_LOW`:
  `EXTRA_LORA_KEY_2`, `EXTRA_LORA_KEY_2_LOW`, …)
- `EXTRA_LORA_URL` / `EXTRA_LORA_URL_LOW` — direct download URLs
  (for files too large for the storage web UI's 500 MB upload limit)
- `EXTRA_LORA_STRENGTH` (+ `_2` / `_3`) — default strengths

The worker downloads the files at boot into ComfyUI's `models/loras/`
and **fails the boot if a configured file cannot be fetched**. If
only one side of a pair is configured, that file is applied to **both**
branches (single-file LoRAs are used that way). Changing LoRAs = edit
the env vars + restart the worker; no rebuild. Note the memo's Wan
LoRAs (Blink series etc.) are exactly this high/low pair shape.

## Graph notes

- Sampler settings are the official templates' standard mode:
  ModelSamplingSD3 shift 5.0 on both branches, KSamplerAdvanced pair
  (high: noise on, steps 0→steps/2, leftover noise on; low: noise off,
  steps/2→end), euler/simple. The templates' LightX2V 4-step mode is
  not used — its LoRAs are not in the slim set.
- **fps handling:** upstream kijai/ComfyUI-MMAudio's sampler slices
  the first N frames of the batch for its CLIP/Synchformer streams,
  which mis-times a 16 fps video (audio ends early). This worker
  installs **phazei/ComfyUI-MMAudio** (pinned commit; his PR #52 to
  kijai), whose sampler resamples the frames uniformly over their
  full span from the frame count + `duration`. The Dockerfile greps
  the cloned tree for that resample and fails the build without it.
  Note this build has no separate `fps` widget — the timing comes
  from passing the true duration with the full frame batch, which is
  what the graph does.
- **Audio windows:** MMAudio was trained on ~8 s clips. Clips up to
  8 s (≤129 frames) use one sampler pass; longer clips are split at
  frame 128 (VHS_SplitImages) into an 8 s window + the remainder,
  sampled separately and joined with the core AudioConcat node. 10 s
  → 8 + 2 s; 15 s → 8 + 7 s.
- The audio branch is omitted entirely when audio is off, so a silent
  job pays no MMAudio load time. VHS_VideoCombine is the output node
  either way (its file lands in the history `gifs` slot, which
  src/comfy_client.py already reads, preferring the final muxed file).
- The cuDNN SDPA opt-out is the sibling workers' boot-written custom
  node (global flag + stripping CUDNN_ATTENTION from
  comfy.ops.SDPA_BACKEND_PRIORITY). There is deliberately **no**
  sitecustomize.py: importing torch before ComfyUI's main.py breaks
  its cudaMallocAsync setup (2026-11-25 lesson).

## First-test watch items (not verifiable without a live ComfyUI)

1. **MMAudio timing:** on the first 5 s audio test, check the worker
   log for the sampler's "Prepared visual streams … duration=5.000s"
   line and confirm the returned MP4's audio runs the full clip
   length. This build derives timing from duration + frame count
   rather than an explicit fps input.
2. **VHS format widgets:** the graph sends only the base
   VHS_VideoCombine inputs (format `video/h264-mp4`); the node's
   optional format widgets (crf, pix_fmt) are left at their defaults
   (19 / yuv420p) instead of being passed as extra API inputs.
3. **I2V-pair T2V quality:** text-to-video runs on the I2V models
   without a start image (the slim-set trade-off). If it looks off,
   the fix is adding the T2V pair to the slim repo (~+28.6 GB), not
   a code change.
4. **BigVGAN path:** the vocoder is expected at
   `models/mmaudio/nvidia/bigvgan_v2_44khz_128band_512x` (baked into
   the image; handler pre-warm as fallback). If the audio stage logs
   a download at job time, this path moved in the pinned build.

## Local checks

```
python3 -m py_compile src/*.py
```

The Dockerfile also runs a build-time graph/validator check (silent
T2V, 5 s audio, 15 s two-window stitching, LoRA pair chains and
fallbacks, validator cases) plus greps asserting the fps-aware
MMAudio build and the VHS node registrations, so a graph-construction
typo fails the image build instead of the first job.
