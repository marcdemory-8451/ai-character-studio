# Spec 03 — Video Mode 1: Keyframe-Driven Video

## Goal
Generate video of a character given:
- **Start frame** (required) — a still of the character (from Mode 2 or uploaded)
- **End frame** (optional) — a target still to interpolate toward
- **Middle frames** (optional) — additional keyframes for multi-section interpolation
- **Text prompt** (optional) — guides motion/action/atmosphere
- **Character video LoRA** (optional) — stacks on the video model for identity retention

## Model selection (user picks one per generation)

| Model | Best for | Min VRAM | Speed | Colab tier |
|---|---|---|---|---|
| **LTX-Video 13B (v0.9.7+)** | Multi-keyframe (start+mid+end); fastest; best default | ~8–10 GB FP8 | ~3 min/5s on A100 | Free/T4 |
| **LTX-Video 2B** | Same capability, lower quality, near real-time | ~6–8 GB | Very fast | Free T4 |
| **Wan2.1-FLF2V-14B** | Best quality two-frame (start+end only); dedicated model | ~14–24 GB GGUF | Slower | A100 |
| **Wan2.2-TI2V-5B** | Clean I2V (start frame only); 720p/24fps | ~24 GB | ~9 min/5s | A100 |
| **FramePack 13B** | Longest clips (up to 60s); start frame only; lowest VRAM | ~6 GB | ~2.5 s/frame | Free T4 |

**Default recommendation:** LTX-Video 13B via ComfyUI for prototyping — Apache 2.0, multi-keyframe,
fits even a T4 (16 GB) at 480p or a free A100 at 720p.

## ComfyUI workflow: `engine/comfyui/workflows/03_video_mode1_ltx.json`

### Node graph (LTX-Video 13B path)

```
[Load Image] start frame
[Load Image] end frame (optional)
[Load Image] middle frame(s) (optional)
    │
    ▼
[LTX-Video KeyframeNode]  ◄─ ComfyUI-LTXVideo official nodes
    │   inputs: images[], timestamps[], strength[]
    │
[LTX-Video ModelLoader]   ◄─ ltx-video-13b-fp8.safetensors
    │
[LTX-Video Sampler]
    │   positive: "<trigger> <user prompt>, cinematic motion"
    │   negative: standard
    │   steps: 30 | cfg: 3.5 | fps: 24 | num_frames: 97 (~4s)
    │
[LTX-Video VAE Decode]
    │
[Video Combine]  ◄─ VHS nodes → .mp4
    │
[Save Video]  outputs/videos/<character>/<timestamp>.mp4
```

For Wan FLF2V path: separate workflow `03_video_mode1_wan_flf2v.json` using WanVideoWrapper
nodes (`WanVideoFLF2VEncode`, `WanVideoSampler`, `WanVideoVAEDecode`).

### Configurable inputs
| Param | Description | Default |
|---|---|---|
| `start_frame_path` | Required | — |
| `end_frame_path` | Optional (null = I2V mode) | null |
| `middle_frames` | List of `{path, timestamp_fraction}` | [] |
| `positive_prompt` | Text describing motion/action | "" |
| `trigger_token` | Prepended to prompt | from library |
| `video_lora_path` | Optional Wan/LTX video LoRA | null |
| `video_lora_strength` | 0–1 | 0.8 |
| `num_frames` | Total output frames | 97 (~4s @ 24fps) |
| `fps` | Output FPS | 24 |
| `width` / `height` | Resolution | 768×512 (LTX default) |
| `seed` | -1 for random | -1 |

### Chaining clips (longer sequences)
To extend beyond a single clip, chain: last frame of clip N → start frame of clip N+1.
The `comfy_client.py` helper exposes `run_chain()` for this.

## Models required
| Model | Source | Local path |
|---|---|---|
| LTX-Video 13B dev FP8 | `Lightricks/LTX-Video` | `models/video/ltxv-13b-0.9.8-dev-fp8.safetensors` |
| LTX-Video 2B distilled FP8 | `Lightricks/LTX-Video` | `models/video/ltxv-2b-0.9.8-distilled-fp8.safetensors` |
| LTX-Video spatial upscaler | `Lightricks/LTX-Video` | `models/video/ltxv-spatial-upscaler-0.9.8.safetensors` |
| Wan2.1-FLF2V-14B (GGUF Q5) | `Wan-AI/Wan2.1-FLF2V-14B-720P` | `models/video/wan21_flf2v_14b_q5.gguf` |

Custom nodes: `ComfyUI-LTXVideo` (official Lightricks), `ComfyUI-VideoHelperSuite` (VHS).

## Validation
- Start/end frames should be visually represented at clip boundaries.
- Character identity should be recognizable through the motion; if it drifts, add a video LoRA.
- No temporal flickering — LTX-Video 13B is generally clean; if flickering occurs, lower cfg (try 3.0).
