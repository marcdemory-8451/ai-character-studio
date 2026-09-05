# Spec 02 — Character Stills Generation

## Goal
Generate controlled still images of a trained character, driven by:
- Text prompt (required)
- Character LoRA (required — from Spec 01)
- Optional reference image (IP-Adapter / InstantID for face consistency)
- Optional pose control (DWPose body + face)
- Optional depth/structure control (Depth Anything V2)

These stills are the source of keyframes for Modes 1, 2, and 3.

## ComfyUI workflow: `engine/comfyui/workflows/02_stills.json`

### Node graph (logical)

```
[Checkpoint Loader] SDXL base
    │
    ├── [LoRA Loader] character LoRA (from library)
    │       │
    │       ▼
    │   [CLIP Text Encode] positive: "ohwx_aria, <user prompt>"
    │   [CLIP Text Encode] negative: standard negative
    │       │
    │       ▼                    ┌─ [Load Image] ref image (optional)
    │   [KSampler]               │       │
    │       │                    │   [IPAdapter+] ◄─ IPAdapter SDXL FaceID model
    │       │◄───────────────────┘
    │       │
    │       │  ┌─ [Load Image] pose source (optional)
    │       │  │       │
    │       │  │   [DWPose Estimator]  ◄─ body+face, 133 keypoints
    │       │  │       │
    │       │  │   [ControlNet Apply] ◄─ SDXL OpenPose ControlNet
    │       │  │       │
    │       │◄─┘       │
    │       │          │  ┌─ [Load Image] depth source (optional)
    │       │          │  │       │
    │       │          │  │   [Depth Anything V2]
    │       │          │  │       │
    │       │          │  │   [ControlNet Apply] ◄─ SDXL depth ControlNet
    │       │◄─────────┴──┘
    │       │
    │   [VAE Decode]
    │       │
    └── [Save Image] outputs/images/<character>/<timestamp>.png
```

### Key nodes (custom)
- `ComfyUI-Impact-Pack` — IP-Adapter+, face detection
- `comfyui_controlnet_aux` — DWPose estimator, Depth Anything V2
- `ComfyUI-Advanced-ControlNet` — multi-ControlNet stacking
- `was-node-suite-comfyui` — filename/path templating

### Configurable inputs (patched before posting to /prompt)
| Param | Node | Description |
|---|---|---|
| `lora_path` | LoRA Loader | character LoRA path |
| `trigger_token` | CLIP Text Encode (pos) | prepended to prompt |
| `positive_prompt` | CLIP Text Encode (pos) | user prompt |
| `ref_image_path` | Load Image (IPAdapter) | face/appearance reference |
| `pose_image_path` | Load Image (pose) | pose source image |
| `depth_image_path` | Load Image (depth) | depth source image |
| `controlnet_strength` | ControlNet Apply | 0–1, default 0.7 |
| `ipadapter_strength` | IPAdapter+ | 0–1, default 0.6 |
| `steps` | KSampler | default 30 |
| `cfg` | KSampler | default 7.0 |
| `seed` | KSampler | -1 for random |
| `width` / `height` | Empty Latent | default 1024×1024 |
| `batch_size` | KSampler | default 4 |

## Models required
| Model | HF/source | Local path |
|---|---|---|
| SDXL base | `stabilityai/stable-diffusion-xl-base-1.0` | `models/checkpoints/sdxl_base.safetensors` |
| SDXL VAE | `madebyollin/sdxl-vae-fp16-fix` | `models/vae/sdxl_vae.safetensors` |
| IP-Adapter+ SDXL FaceID | `h94/IP-Adapter-FaceID` | `models/ipadapter/ip-adapter-faceid-plusv2_sdxl.bin` + `ip-adapter-faceid-plusv2_sdxl_lora.safetensors` (both required) |
| SDXL ControlNet OpenPose | `thibaud/controlnet-openpose-sdxl-1.0` | `models/controlnet/sdxl_openpose.safetensors` |
| SDXL ControlNet Depth | `diffusers/controlnet-depth-sdxl-1.0` | `models/controlnet/sdxl_depth.safetensors` |
| DWPose | auto-downloaded by comfyui_controlnet_aux | `models/dwpose/` |
| Depth Anything V2 | auto-downloaded by comfyui_controlnet_aux | `models/depth_anything/` |

## Python helper (Option A thin layer)
```python
from engine.comfyui.setup.comfy_client import ComfyClient
from library.library import CharacterLibrary

lib = CharacterLibrary()
char = lib.get_character_by_name("Aria")
client = ComfyClient("http://localhost:8188")

output = client.run_workflow(
    "engine/comfyui/workflows/02_stills.json",
    params={
        "lora_path": char["lora_path"],
        "trigger_token": char["trigger"],
        "positive_prompt": "portrait, dramatic lighting, forest background",
        "pose_image_path": "reference_pose.jpg",
        "batch_size": 4,
    }
)
lib.log_generation("still", output["path"], character_id=char["id"], prompt=output["prompt"])
```

## Validation
- Characters should be recognizable vs training refs at seed variety.
- Pose ControlNet: skeleton extracted from the source image should match body position in output.
- Face should hold with IP-Adapter; increase `ipadapter_strength` (up to 0.8) if drifting.
