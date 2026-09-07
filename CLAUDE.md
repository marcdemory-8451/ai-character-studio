# AI Character Video Studio

Self-hosted pipeline for generating images and video of original fictional characters.
Option A implementation: pure-diffusers Colab-native generation, single-user, local + Colab.
(Originally ComfyUI-centric; that route was superseded in Phase 1 — see the Colab notebooks
section below.)

## Project structure

```
engine/comfyui/workflows/   parameterized ComfyUI workflow JSONs (one per mode)
engine/comfyui/setup/       install scripts + comfy_client.py API helper
engine/training/            ai-toolkit SDXL configs + musubi-tuner Wan configs
library/                    SQLite character DB + library.py CRUD helpers
agent/                      Mode 3 orchestrator (agent.py, scene_planner.py, llm_provider.py)
prototype/                  Colab notebooks (00-05): 01a/01c training, 02a/03a/03b/04a/04b/05
                            generation (active); 00/01b/02/03/04 are the deprecated ComfyUI route
specs/                      Per-mode implementation specs (00-05)
characters/<name>/          Per-character data: reference-images, captions, loras, samples
outputs/                    Generated images and videos
```

## Key models in use (all Apache 2.0 or RAIL-M)

| Role | Model | VRAM |
|---|---|---|
| Image base | FLUX.1-dev (guidance-distilled) | ~24 GB BF16 |
| Face consistency | flux-ip-adapter (XLabs) — optional | inference |
| Pose control | DWPose (via comfyui_controlnet_aux) | inference |
| Video keyframe FLF2V | LTX-Video 13B (multi-keyframe) | ~8–10 GB FP8 |
| Video FLF2V quality | Wan2.1-FLF2V-14B | ~14–24 GB |
| Character replacement | Wan2.2-Animate-14B (Mix mode) | ~22–24 GB |
| Video editing | Wan VACE 1.3B / 14B | ~8–24 GB |
| Upscale | SeedVR2 3B | ~16 GB |
| Image LoRA training | ai-toolkit (ostris) | ~10–16 GB |
| Video LoRA training | musubi-tuner (kohya) | ~12–24 GB |

## Generation engine (Colab-native diffusers)

Generation runs directly in Colab notebooks using diffusers pipelines — no ComfyUI server, no
tunnel, no workflow JSONs. Each generation block is one notebook (`prototype/02a` … `05`);
model loading is VRAM-tiered (resident / group-offload / sequential-cpu-offload) and resumable
states persist to Drive. The old ComfyUI route (`engine/comfyui/`, `comfy_client.py`) is kept
for reference and the library browser only, not the prototype pipeline.

## Character library

Characters live in `library/characters.db` (SQLite) + `characters/<name>/` folder tree.
CRUD via `library/library.py:CharacterLibrary`. One character = one FLUX.1-dev LoRA (required,
trained via `01c`, ai-toolkit) + optional video LoRA (Wan / LTX).

## Agent (Mode 3)

`agent/agent.py:Agent.run(character_name, prompt)` — state machine, resumable on failure.
LLM provider is swappable: Ollama (default, local Qwen/Llama) or Claude (set `ANTHROPIC_API_KEY`).

## Colab notebooks

**Generation route: ComfyUI is superseded.** The headless-server + tunnel + workflow-JSON route
proved fragile, so generation is now pure-diffusers Colab-native (no server, no tunnel, no custom
nodes). The old ComfyUI notebooks (`00`, `02`, `03`, `04`) are kept for reference but
**deprecated — use the new numbered set below**. `engine/comfyui/` remains for the library browser
and any future ComfyUI revival, not for the prototype pipeline.

| Notebook | Purpose | Route | GPU |
|---|---|---|---|
| `prototype/00_phase0_comfyui_setup.ipynb` | ~~ComfyUI + nodes + models~~ | ⚠️ deprecated | A100 |
| `prototype/01a_caption_refs.ipynb` | Auto-caption reference images | active | T4 |
| `prototype/01b_train_sdxl_lora.ipynb` | ~~Train SDXL LoRA~~ (superseded by 01c) | ⚠️ deprecated | A100 |
| `prototype/01c_train_flux_lora.ipynb` | Train FLUX.1-dev character LoRA (ai-toolkit, uv venv) | active | A100 40GB+ |
| `prototype/02_test_stills.ipynb` | ~~Stills via ComfyUI API~~ | ⚠️ deprecated | A100 |
| `prototype/02a_test_stills_flux.ipynb` | FLUX.1-dev stills + LoRA, VRAM-tiered, commented alternates (Redux/Kontext/schnell/fill) | active | A100 |
| `prototype/03_test_video_mode1.ipynb` | ~~LTX via ComfyUI~~ | ⚠️ deprecated | A100 |
| `prototype/03a_test_video_mode1_ltx.ipynb` | LTX-Video 0.9.8-13B multi-keyframe FLF2V + latent upscale, resumable, commented alternates (2B/GGUF/fp8) | active | A100 |
| `prototype/03b_test_video_mode1_wan_flf2v.ipynb` | Wan2.1-FLF2V-14B 720p first/last-frame (quality tier), resumable | active | A100 |
| `prototype/04_test_video_mode2.ipynb` | ~~Wan Animate via ComfyUI~~ | ⚠️ deprecated | A100 80GB |
| `prototype/04a_test_video_mode2_animate.ipynb` | Wan2.2-Animate-14B character replacement (official preprocessing venv + diffusers pipeline) | active | A100 80GB |
| `prototype/04b_test_video_mode2_vace.ipynb` | Wan VACE 1.3B swap-anything (SAM2-tracked mask + reference image) | active | A100 40GB+ |
| `prototype/05_agentic_video.ipynb` | Mode 3: resumable state machine, local Ollama LLM (Claude API commented), FLUX keyframes → LTX clips → ffmpeg stitch | active | A100 |

**Shared notebook conventions** (all active notebooks): mount Drive → `DRIVE_BASE =
/content/drive/MyDrive/ai_character_studio`; **models download straight to local `/content`
(e.g. `/content/flux_dev`, `/content/ltx_13b`) via resumable `snapshot_download` — NO Drive
mirror of models and `HF_HOME` on local `/content/hf_cache`**, because a Drive-backed HF cache
corrupts large gated models (FUSE atomic-rename) and the `cp -rn` Drive mirror produced
partial/corrupt copies that broke loads (re-downloading each session is the deliberate trade);
HF_TOKEN from Colab Secrets; `uv pip install --system` for inference (isolated venv only
for 01c/04a where the repo pins its own deps); long logs/downloads to files, never an unread
pipe; outputs + `metadata.json` logging to Drive (small files — safe).

## License reminder

- DWPose (Apache 2.0) is the pose estimator — do NOT use OpenPose (CMU non-commercial).
- SMPL is non-commercial — avoid models that require it unless you have a Meshcapade license.
- Wan 2.x, LTX-Video, SDXL are Apache 2.0 / RAIL-M — commercial-safe.
- HunyuanVideo is geo-restricted (no EU/UK/KR) — not in the default stack.

## Phase progress

- [x] Phase 0 — Environment (install scripts + Colab setup notebook)
- [x] Phase 1 scaffolded — validation notebooks ready to run
- [ ] Phase 1 — Actually run and validate each block (needs GPU)
- [x] FLUX switch (iteration 2) — SDXL→FLUX.1-dev character LoRA; SDXL route kept but deprecated
- [x] ComfyUI route replaced — pure-diffusers Colab-native generation notebooks (02a–05)
- [ ] Phase 2 — Thin library browser (Gradio app)
- [x] Phase 3 — FLUX vs SDXL decision: **FLUX wins** (realism/likeness/steerability), decided
      pre-validation; SDXL kept as the low-VRAM/cheap fallback
- [ ] Phase 4 — Option B product architecture
