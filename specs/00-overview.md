# Implementation Specs — Overview

Option A: ComfyUI-centric prototype. ComfyUI is the generation engine for all inference;
ai-toolkit and musubi-tuner handle training as separate jobs. Everything is driven via
ComfyUI's `/prompt` HTTP API or saved workflow UIs.

## Mode → Spec file map

| Mode | Spec | Status |
|---|---|---|
| Character Creation (training) | `01-character-creation.md` | draft |
| Character Stills Generation | `02-stills-generation.md` | draft |
| Video Mode 1 — Keyframe-driven | `03-video-mode1-keyframes.md` | draft |
| Video Mode 2 — Character replacement | `04-video-mode2-replacement.md` | draft |
| Video Mode 3 — Agentic | `05-video-mode3-agentic.md` | draft |

## Shared conventions

- All paths are relative to the project root (`video/`).
- Character IDs are uuid4; library is `library/characters.db` (SQLite).
- ComfyUI runs on `http://localhost:8188` locally, same port on Colab (exposed via ngrok or
  cloudflared tunnel).
- Workflow inputs are injected by patching the JSON `inputs` dict before posting to `/prompt`.
  See `engine/comfyui/setup/comfy_client.py` for the helper.
- All models use Apache 2.0 or RAIL-M licenses unless explicitly noted.

## Data flow summary

```
Reference images
    │
    ▼
[01] ai-toolkit training ──► character LoRA  ──► library/characters.db
                                                        │
    ┌───────────────────────────────────────────────────┘
    │
    ▼
[02] Stills: SDXL + LoRA + DWPose/depth/IP-Adapter ──► .png → outputs/images/
    │
    ▼ (first/last keyframes)
[03] Video Mode 1: LTX-Video 13B (multi-keyframe) or Wan FLF2V ──► .mp4
[04] Video Mode 2: DWPose extract → Wan2.2-Animate-14B Mix ──► .mp4
[05] Video Mode 3: LLM → scene plan → [02] keyframes → [03] interpolate → ffmpeg ──► .mp4
```
