# AI Character Video Studio

Self-hosted pipeline for generating images and video of original fictional characters.
Option A implementation: ComfyUI-centric, single-user, local + Colab.

## Project structure

```
engine/comfyui/workflows/   parameterized ComfyUI workflow JSONs (one per mode)
engine/comfyui/setup/       install scripts + comfy_client.py API helper
engine/training/            ai-toolkit SDXL configs + musubi-tuner Wan configs
library/                    SQLite character DB + library.py CRUD helpers
agent/                      Mode 3 orchestrator (agent.py, scene_planner.py, llm_provider.py)
prototype/                  Colab notebooks (00-04) for each building block
specs/                      Per-mode implementation specs (00-05)
characters/<name>/          Per-character data: reference-images, captions, loras, samples
outputs/                    Generated images and videos
```

## Key models in use (all Apache 2.0 or RAIL-M)

| Role | Model | VRAM |
|---|---|---|
| Image base | SDXL 1.0 | train: 10–16 GB |
| Face consistency | IP-Adapter FaceID Plus v2 SDXL | inference |
| Pose control | DWPose (via comfyui_controlnet_aux) | inference |
| Video keyframe FLF2V | LTX-Video 13B (multi-keyframe) | ~8–10 GB FP8 |
| Video FLF2V quality | Wan2.1-FLF2V-14B | ~14–24 GB |
| Character replacement | Wan2.2-Animate-14B (Mix mode) | ~22–24 GB |
| Video editing | Wan VACE 1.3B / 14B | ~8–24 GB |
| Upscale | SeedVR2 3B | ~16 GB |
| Image LoRA training | ai-toolkit (ostris) | ~10–16 GB |
| Video LoRA training | musubi-tuner (kohya) | ~12–24 GB |

## ComfyUI engine

ComfyUI runs headless at `http://localhost:8188` (local) or behind a cloudflared tunnel (Colab).
Workflows are posted via `engine/comfyui/setup/comfy_client.py` (`ComfyClient.run_workflow()`).
All inference is parameterized by patching workflow JSON before submission.

## Character library

Characters live in `library/characters.db` (SQLite) + `characters/<name>/` folder tree.
CRUD via `library/library.py:CharacterLibrary`. One character = one SDXL LoRA (required) +
optional Wan 2.2 video LoRA.

## Agent (Mode 3)

`agent/agent.py:Agent.run(character_name, prompt)` — state machine, resumable on failure.
LLM provider is swappable: Ollama (default, local Qwen/Llama) or Claude (set `ANTHROPIC_API_KEY`).

## Colab notebooks (run in order to validate each block)

| Notebook | Purpose | GPU |
|---|---|---|
| `prototype/00_phase0_comfyui_setup.ipynb` | Install ComfyUI + nodes + download models | A100 |
| `prototype/01a_caption_refs.ipynb` | Auto-caption reference images | T4 |
| `prototype/01b_train_sdxl_lora.ipynb` | Train SDXL character LoRA | A100 |
| `prototype/02_test_stills.ipynb` | Test stills via ComfyUI API | A100 |
| `prototype/03_test_video_mode1.ipynb` | Test LTX-Video keyframe interpolation | A100 |
| `prototype/04_test_video_mode2.ipynb` | Test Wan Animate character replacement | A100 80GB |

## License reminder

- DWPose (Apache 2.0) is the pose estimator — do NOT use OpenPose (CMU non-commercial).
- SMPL is non-commercial — avoid models that require it unless you have a Meshcapade license.
- Wan 2.x, LTX-Video, SDXL are Apache 2.0 / RAIL-M — commercial-safe.
- HunyuanVideo is geo-restricted (no EU/UK/KR) — not in the default stack.

## Phase progress

- [x] Phase 0 — Environment (install scripts + Colab setup notebook)
- [x] Phase 1 scaffolded — validation notebooks ready to run
- [ ] Phase 1 — Actually run and validate each block (needs GPU)
- [ ] Phase 2 — Thin library browser (Gradio app)
- [ ] Phase 3 — FLUX vs SDXL decision based on Phase 1 results
- [ ] Phase 4 — Option B product architecture
