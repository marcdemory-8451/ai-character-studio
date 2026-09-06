# Implementation Specs — Overview

Option A: pure-diffusers Colab-native prototype.

> **Note (2026-09):** the original ComfyUI-centric engine described below is **superseded**.
> Generation now runs directly in Colab notebooks using diffusers pipelines (no ComfyUI server,
> tunnel, or workflow JSONs): `prototype/02a` (FLUX stills), `03a`/`03b` (LTX / Wan FLF2V),
> `04a`/`04b` (Wan-Animate / VACE), `05` (agentic). The base model also moved SDXL →
> FLUX.1-dev (iteration 2; train via `prototype/01c`). The specs below still describe the per-mode
> logic (inputs, prompts, fallbacks, data flow) and remain the design of record — only the
> execution substrate (ComfyUI API) was replaced. `engine/comfyui/` is kept for reference only.

Option A (original): ComfyUI was the generation engine for all inference;
ai-toolkit and musubi-tuner handle training as separate jobs. Everything was driven via
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
- Generation runs in Colab via diffusers pipelines (see note above). Every generation notebook
  follows the same header pattern: Drive mount → `DRIVE_BASE`, `HF_HOME` on Drive, HF_TOKEN from
  Colab Secrets, `uv pip install --system`, VRAM-tiered model loading, long logs to files,
  outputs + `metadata.json` to Drive.
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
