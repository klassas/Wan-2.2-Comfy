"""ComfyUI node graph for Wan 2.2 + NSFW MMAudio, built in code per job.

One graph, two stages in a single ComfyUI process (no intermediate
file round-trip):

    UNETLoader (high noise) ─[LoRA pair, high files]─> ModelSamplingSD3 ─┐
    UNETLoader (low noise)  ─[LoRA pair, low files]──> ModelSamplingSD3 ─┤
    CLIPLoader (umt5, type wan) ─> CLIPTextEncode x2                     │
    WanImageToVideo (start frame optional) ─> KSamplerAdvanced (high,    │
      steps 0..steps/2, leftover noise on) ─> KSamplerAdvanced (low,      │
      steps/2..end) ─> VAEDecode ─> frames ─┬─> VHS_VideoCombine         │
                                            └─> MMAudioSampler(s) ───────┘
                                                 (audio input of combine)

Video stage follows ComfyUI's official Wan 2.2 templates
(Comfy-Org/workflow_templates, video_wan2_2_14B_i2v / _t2v) in their
LightX2V 4-step mode: the LightX2V I2V LoRA pair bundled in the slim
set is applied unconditionally on both branches (strength 1.0),
ModelSamplingSD3 shift 5.0, 8 steps split evenly between the
high-noise and low-noise models, cfg 1.0, euler/simple. The slim
model set carries only the I2V pair; WanImageToVideo without a
start_image runs it as T2V, which is the documented slim-set
trade-off (see WAN_MMAUDIO_RESEARCH.md).
The templates use no CLIP-vision branch, so none is needed here.

Audio stage (kijai/ComfyUI-MMAudio, phazei's fps-aware build — PR #52
to kijai, "Scale input frames to match CLIP and synchformer fps"):
its MMAudioSampler resamples the frames it is given uniformly over
their full span from the frame count + `duration`, so a 16 fps Wan
clip is timed correctly as long as `duration` is the clip's real
duration. MMAudio was trained on ~8 s clips, so longer clips are cut
into <= 8 s windows: VHS_SplitImages slices the frame batch,
one MMAudioSampler runs per window, and core AudioConcat joins the
pieces in order. With audio off (or mute) the whole branch is omitted
and VHS_VideoCombine writes a silent MP4 through the same output path.

Node schemas verified against ComfyUI v0.46.2 sources
(comfy_extras/nodes_wan.py WanImageToVideo, nodes_model_advanced.py
ModelSamplingSD3, nodes_audio.py AudioConcat, nodes.py
KSamplerAdvanced), phazei/ComfyUI-MMAudio master nodes.py
(MMAudioModelLoader / MMAudioFeatureUtilsLoader / MMAudioSampler) and
ComfyUI-VideoHelperSuite videohelpersuite/nodes.py +
image_latent_nodes.py (VHS_VideoCombine / VHS_SplitImages).
"""

# Model files, as laid out in the slim repo (the Model Cache snapshot
# is symlinked into these exact category dirs by model_link.py).
UNET_HIGH_NAME = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
UNET_LOW_NAME = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
CLIP_NAME = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_NAME = "wan_2.1_vae.safetensors"
MMAUDIO_MODEL_NAME = "mmaudio_large_44k_nsfw_gold_8.5k_final_fp16.safetensors"
MMAUDIO_VAE_NAME = "mmaudio_vae_44k_fp16.safetensors"
MMAUDIO_SYNCHFORMER_NAME = "mmaudio_synchformer_fp16.safetensors"
MMAUDIO_CLIP_NAME = "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"
# LightX2V 4-step acceleration LoRAs (Comfy-Org repackaged v1 pair),
# bundled in the slim set under loras/ and always applied (strength 1).
LIGHTX2V_HIGH_NAME = "wan_lightx2v_high.safetensors"
LIGHTX2V_LOW_NAME = "wan_lightx2v_low.safetensors"

OUTPUT_NODE_ID = "combine"

FPS = 16
# One MMAudio window: 8 s at 16 fps = 128 frames (see config).
AUDIO_WINDOW_FRAMES = 128
AUDIO_WINDOW_SECONDS = 8.0


def _model_branch(nodes: dict, tag: str, unet_name: str,
                  extra_loras: list | None, side: int, shift: float) -> list:
    """UNETLoader -> fixed LightX2V LoRA -> this branch's LoRA files
    -> ModelSamplingSD3.

    extra_loras items are ((high_name, low_name), strength); `side` is
    0 for the high-noise branch, 1 for the low-noise branch. A slot
    whose file for this side is missing falls back to the other side's
    file (single-file LoRAs apply to both branches); a slot with both
    names None is skipped entirely.
    """
    nodes[f"unet_{tag}"] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
    }
    model_ref = [f"unet_{tag}", 0]
    lightx2v_name = LIGHTX2V_HIGH_NAME if side == 0 else LIGHTX2V_LOW_NAME
    nodes[f"lora_{tag}_lightx2v"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": model_ref,
            "lora_name": lightx2v_name,
            "strength_model": 1.0,
        },
    }
    model_ref = [f"lora_{tag}_lightx2v", 0]
    for i, item in enumerate(extra_loras or []):
        if not item:
            continue  # slot not configured (or disabled for this job)
        pair, strength = item
        if not pair:
            continue
        name = pair[side] or pair[1 - side]
        if not name:
            continue
        node_id = f"lora_{tag}" if i == 0 else f"lora_{tag}_{i + 1}"
        nodes[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": name,
                "strength_model": strength,
            },
        }
        model_ref = [node_id, 0]
    nodes[f"sampling_{tag}"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": model_ref, "shift": shift},
    }
    return [f"sampling_{tag}", 0]


def _audio_windows(frames: int) -> list[tuple[int, float]]:
    """Split `frames` (16 fps, 4n+1 grid) into MMAudio windows.

    Returns [(frame_count, duration_seconds), ...] in order. Every
    window but the last spans exactly AUDIO_WINDOW_FRAMES frames /
    AUDIO_WINDOW_SECONDS; the last takes the remainder, its duration
    being the clip time it actually covers.
    """
    total_seconds = (frames - 1) / FPS
    windows: list[tuple[int, float]] = []
    start = 0
    while start < frames:
        count = min(AUDIO_WINDOW_FRAMES, frames - start)
        duration = min(AUDIO_WINDOW_SECONDS, total_seconds - start / FPS)
        windows.append((count, round(duration, 4)))
        start += AUDIO_WINDOW_FRAMES
    return windows


def _audio_branch(nodes: dict, *, frames: int, audio_seed: int,
                  audio_prompt: str, audio_negative_prompt: str,
                  audio_steps: int, audio_cfg: float) -> list:
    """Adds the MMAudio loaders + per-window samplers (+ AudioConcat
    when the clip needs more than one window) and returns the AUDIO
    ref for VHS_VideoCombine. Frames come from the `decode` node."""
    nodes["mmaudio_model_loader"] = {
        "class_type": "MMAudioModelLoader",
        "inputs": {"mmaudio_model": MMAUDIO_MODEL_NAME, "base_precision": "fp16"},
    }
    nodes["mmaudio_feature_utils"] = {
        "class_type": "MMAudioFeatureUtilsLoader",
        "inputs": {
            "vae_model": MMAUDIO_VAE_NAME,
            "synchformer_model": MMAUDIO_SYNCHFORMER_NAME,
            "clip_model": MMAUDIO_CLIP_NAME,
            "mode": "44k",
            "precision": "fp16",
        },
    }
    windows = _audio_windows(frames)
    frames_ref: list = ["decode", 0]
    audio_ref: list | None = None
    for wi, (count, duration) in enumerate(windows):
        if wi < len(windows) - 1:
            split_id = f"split_frames_{wi + 1}"
            nodes[split_id] = {
                "class_type": "VHS_SplitImages",
                "inputs": {"images": frames_ref, "split_index": count},
            }
            window_ref = [split_id, 0]  # IMAGE_A
            frames_ref = [split_id, 2]  # IMAGE_B = the remaining frames
        else:
            window_ref = frames_ref
        sampler_id = "audio_sampler" if len(windows) == 1 else f"audio_sampler_{wi + 1}"
        nodes[sampler_id] = {
            "class_type": "MMAudioSampler",
            "inputs": {
                "mmaudio_model": ["mmaudio_model_loader", 0],
                "feature_utils": ["mmaudio_feature_utils", 0],
                "duration": duration,
                "steps": audio_steps,
                "cfg": audio_cfg,
                "seed": audio_seed + wi,
                "prompt": audio_prompt,
                "negative_prompt": audio_negative_prompt,
                "mask_away_clip": False,
                "force_offload": True,
                "images": window_ref,
            },
        }
        if audio_ref is None:
            audio_ref = [sampler_id, 0]
        else:
            concat_id = f"audio_concat_{wi}"
            nodes[concat_id] = {
                "class_type": "AudioConcat",
                "inputs": {
                    "audio1": audio_ref,
                    "audio2": [sampler_id, 0],
                    "direction": "after",
                },
            }
            audio_ref = [concat_id, 0]
    assert audio_ref is not None  # windows is never empty (frames >= 5)
    return audio_ref


def build_workflow(
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 960,
    height: int = 544,
    frames: int = 81,
    seed: int = 0,
    steps: int = 8,
    cfg: float = 1.0,
    shift: float = 5.0,
    start_image_name: str | None = None,
    extra_loras: list | None = None,
    audio: bool = True,
    audio_prompt: str = "",
    audio_negative_prompt: str = "",
    audio_steps: int = 25,
    audio_cfg: float = 4.5,
    audio_seed: int | None = None,
    filename_prefix: str = "wan22/job",
) -> dict:
    """Returns the API-format graph: {node_id: {"class_type", "inputs"}}.

    start_image_name: a file already uploaded to ComfyUI's input/
    directory, or None for text-to-video. `frames` must already be on
    the 4n+1 grid (the validator snaps it). extra_loras items are
    ((high_name, low_name), strength) pairs — see _model_branch.
    """
    nodes: dict[str, dict] = {
        "clip_loader": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP_NAME, "type": "wan", "device": "default"},
        },
        "vae_loader": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
        "encode_pos": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip_loader", 0], "text": prompt},
        },
        "encode_neg": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip_loader", 0], "text": negative_prompt},
        },
    }
    high_ref = _model_branch(nodes, "high", UNET_HIGH_NAME, extra_loras, 0, shift)
    low_ref = _model_branch(nodes, "low", UNET_LOW_NAME, extra_loras, 1, shift)

    i2v_inputs: dict = {
        "positive": ["encode_pos", 0],
        "negative": ["encode_neg", 0],
        "vae": ["vae_loader", 0],
        "width": width,
        "height": height,
        "length": frames,
        "batch_size": 1,
    }
    if start_image_name:
        nodes["load_start"] = {
            "class_type": "LoadImage",
            "inputs": {"image": start_image_name},
        }
        i2v_inputs["start_image"] = ["load_start", 0]
    nodes["i2v"] = {"class_type": "WanImageToVideo", "inputs": i2v_inputs}

    split_step = steps // 2
    nodes["sampler_high"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": high_ref,
            "add_noise": "enable",
            "noise_seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["i2v", 0],
            "negative": ["i2v", 1],
            "latent_image": ["i2v", 2],
            "start_at_step": 0,
            "end_at_step": split_step,
            "return_with_leftover_noise": "enable",
        },
    }
    nodes["sampler_low"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": low_ref,
            "add_noise": "disable",
            "noise_seed": 0,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["i2v", 0],
            "negative": ["i2v", 1],
            "latent_image": ["sampler_high", 0],
            "start_at_step": split_step,
            "end_at_step": 10000,
            "return_with_leftover_noise": "disable",
        },
    }
    nodes["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler_low", 0], "vae": ["vae_loader", 0]},
    }

    combine_inputs: dict = {
        "images": ["decode", 0],
        "frame_rate": FPS,
        "loop_count": 0,
        "filename_prefix": filename_prefix,
        "format": "video/h264-mp4",
        "pingpong": False,
        "save_output": True,
    }
    if audio:
        combine_inputs["audio"] = _audio_branch(
            nodes,
            frames=frames,
            audio_seed=seed if audio_seed is None else audio_seed,
            audio_prompt=audio_prompt,
            audio_negative_prompt=audio_negative_prompt,
            audio_steps=audio_steps,
            audio_cfg=audio_cfg,
        )
    nodes[OUTPUT_NODE_ID] = {
        "class_type": "VHS_VideoCombine",
        "inputs": combine_inputs,
    }
    return nodes
