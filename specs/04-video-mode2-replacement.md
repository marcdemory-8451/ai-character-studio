# Spec 04 — Video Mode 2: Character Replacement

## Goal
Replace a subject/character in an existing video clip with a chosen character from the library,
matching the original pose, movement, and expression while preserving background and timing.

## Pipeline

```
Source video
    │
    ├── [Step 1] DWPose extraction (per-frame skeleton)
    │
    ├── [Step 2a] Wan2.2-Animate-14B (Mix/Replacement mode)   ← primary path
    │       Inputs: source video + character reference image
    │       Outputs: replaced video (background preserved, identity swapped)
    │
    └── [Step 2b] Wan VACE Swap-Anything (alternate path)
            Inputs: source video + SAM 2 mask + character reference image
            Outputs: masked region regenerated with character
    │
    ▼
[Step 3] SeedVR2 upscale/restore (optional, final pass)
    │
    ▼
Output .mp4
```

## Step 1 — Pre-processing: DWPose extraction

Pre-process the source video before sending to ComfyUI. This creates the skeleton overlay
video that guides the replacement.

Script: `engine/training/captioning/extract_dwpose.py`

```python
# extract_dwpose.py (outline)
# For each frame: run DWPose → draw skeleton → save as frame PNG
# Output: dwpose_frames/ folder + dwpose_video.mp4
```

ComfyUI alternative: `comfyui_controlnet_aux` `DWPose Estimator (Video)` node processes the
loaded video directly frame-by-frame inside the graph.

## Step 2a — Wan2.2-Animate-14B: Character Replacement (primary)

**ComfyUI workflow:** `engine/comfyui/workflows/04_video_mode2_animate.json`

This uses Wan2.2-Animate in its **Mix (Replacement) mode** — it takes:
- A source video (the original clip with the target subject)
- A character reference image (canonical still of your fictional character)

...and replaces the subject in the source video with your character, auto-matching lighting,
background, and pose timing.

### Node graph

```
[VHS Load Video] source_video.mp4
    │
[Load Image] character_reference.png  ← from Mode 2 stills or uploaded
    │
[WanVideoModelLoader]  ◄─ Wan2.2-Animate-14B weights
    │
[WanVideoAnimateEncode]
    │   mode: "replacement"   (Mix mode)
    │   ref_image: character reference
    │   source_video: loaded frames
    │
[WanVideoSampler]
    │   steps: 20 (with 4-step LoRA acceleration) or 30 (standard)
    │   cfg: 7.0
    │
[WanVideoVAEDecode]
    │
[VHS Video Combine] → .mp4
```

### Key parameters
| Param | Description | Default |
|---|---|---|
| `source_video_path` | Input video to replace character in | — |
| `reference_image_path` | Character still (canonical reference) | — |
| `video_lora_path` | Optional Wan video LoRA for stronger identity | null |
| `video_lora_strength` | 0–1 | 0.85 |
| `steps` | Denoising steps (20 with accel LoRA, 30 standard) | 20 |
| `max_frames` | Clip to N frames if source is long (memory) | 97 |
| `offload_model` | Enable for <48 GB VRAM (slower) | true |

VRAM: ~22–24 GB FP8 with offload. Needs an A100 on Colab.

## Step 2b — Wan VACE Swap-Anything (alternate / lower-VRAM path)

Use this when: source clip has complex multi-subject scenes (need precise masking),
or when you're on a 16 GB card and need the 1.3B model.

**ComfyUI workflow:** `engine/comfyui/workflows/04_video_mode2_vace.json`

1. Load source video.
2. **SAM 2** mask node: track the subject across all frames → binary mask video.
3. **WanVideoVACEEncode** in `swap_anything` mode: source video + mask + reference image.
4. **WanVideoSampler** → decode → export.

VRAM: ~8 GB (1.3B model, 480p) or ~16–24 GB (14B, 720p).

## Step 3 — SeedVR2 upscale (optional)

**ComfyUI workflow:** append SeedVR2 node after video combine.
Node: `SeedVR2VideoUpscale` (ComfyUI-SeedVR2 node)
VRAM: ~16 GB; can run sequentially if memory is shared.

## Models required
| Model | Source | Local path |
|---|---|---|
| Wan2.2-Animate-14B | `Wan-AI/Wan2.2-Animate-14B` | `models/video/wan22_animate_14b_fp8.safetensors` |
| Wan VACE 1.3B | `Wan-AI/Wan2.1-VACE-1.3B` | `models/video/Wan2.1-VACE-1.3B/` (snapshot dir, diffusion_pytorch_model.safetensors inside) |
| SAM 2 | `facebook/sam2` | auto-downloaded by ComfyUI-SAM2 node |
| SeedVR2 3B | `ByteDance-Seed/SeedVR` | `models/upscale/seedvr2_3b.safetensors` |

Custom nodes: `ComfyUI-WanVideoWrapper` (Kijai), `ComfyUI-SAM2` (kijai or saltai),
`ComfyUI-VideoHelperSuite`, `ComfyUI-SeedVR2`.

## Validation
- Background of source clip should be preserved (check static elements frame-by-frame).
- Character's pose should match original subject's pose (compare skeleton overlay).
- Identity should match the character reference image (especially face).
- No background bleed-through on character edges (if present: add feathering on the mask).
