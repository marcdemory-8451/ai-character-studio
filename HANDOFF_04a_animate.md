# Handoff — 04a: Wan2.2-Animate-14B character replacement

**Goal:** Take an AI character reference still + a source video and re-render the source
person AS that character (replace mode), preserving the original scene/pose/motion, using
pure-`diffusers` `WanAnimatePipeline` on Google Colab (A100).

## Links
- **Repo:** https://github.com/marcdemory-8451/ai-character-studio (branch `main`)
- **Local clone:** `/Users/m626288/CLAUDE_PROJECTS/ai-character-studio`
- **Colab (open from GitHub):** https://colab.research.google.com/github/marcdemory-8451/ai-character-studio/blob/main/prototype/04a_test_video_mode2_animate.ipynb
- **Notebook file:** `prototype/04a_test_video_mode2_animate.ipynb`
- **Models:** `Wan-AI/Wan2.2-Animate-14B` (preprocessing weights, only `process_checkpoint/*` needed) · `Wan-AI/Wan2.2-Animate-14B-Diffusers` (inference)

## Architecture (how the notebook is structured)
1. **Config + mount Drive** — sets `CHARACTER_NAME='Yuna'`, `DRIVE_BASE`, `VID_OUT`, `HF_HOME=/content/hf_cache`.
2. **Upload inputs** — `files.upload()` for SOURCE VIDEO (.mp4) + CHARACTER REFERENCE still.
3. **Install** — clones `Wan-Video/Wan2.2`, builds an **isolated uv venv** (`/content/Wan2.2/.venv`) for the OFFICIAL `preprocess_data.py` (keeps Colab's torch intact); installs the diffusers stack into the **system** env for inference.
4. **Preprocessing** — downloads `process_checkpoint/*`, runs `preprocess_data.py --replace_flag` → produces `src_face.mp4`, `src_pose.mp4`, `src_bg.mp4`, `src_mask.mp4`.
5. **(Review)** — montage of the extracted mask.
6. **Load pipeline + run** — `WanAnimatePipeline.from_pretrained("...-Diffusers")`, VAE fp32, transformer bf16, `mode="replace"`, group-offload on A100-40 / resident on A100-80.
7. Metadata log. 8. Commented alternates (animate mode, relighting LoRA, CFG, LightX2V, own-mask).

## Current status (2026-09-20)
**Working:**
- Cells 1–2 (config, upload) ✅ on the current runtime. Uploaded inputs: `walking-down-street (1).mp4` (source) + `reference (1).png` (Yuna still). *(Test inputs; source is a Pixabay clip, ref is a Yuna generation.)*
- Cell 3 install ✅ — venv builds, torch 2.6.0+cu124 in venv, SAM2/decord/peft/etc. install, system diffusers from git main, `cuda True`.
- Cell 4 download ✅ — `snapshot_download` pulled `process_checkpoint/*` (3.98 GB) successfully.

**Last known failure (now fixed, needs verify):**
- Preprocessing (`preprocess_data.py`) exited 1 with `ModuleNotFoundError: No module named 'moviepy'`. `moviepy` is imported by `process_pipepline.py` but is NOT in Wan2.2's requirements. **Fix pushed** (commit `3808421`): added `moviepy imageio-ffmpeg` to the venv install.

**Immediate next step:** reload the Colab tab to pull `3808421`, re-run cell 3 (venv rebuild now includes moviepy), then cell 4. Expect the 4 `src_*.mp4` outputs. Then run cells 5→6 (first full inference has never run yet — that's the real unknown to validate: VRAM/offloading + `mode="replace"` output quality).

## Fixes already applied (all on `main`)
| Commit | Fix |
|---|---|
| `1d922b0` | venv install: torch **2.6.0**/torchvision 0.21.0 on **cu124** (2.7.1 doesn't exist on cu124); drop `flash_attn` (inference-only, preprocess doesn't need it); add animate reqs |
| `accafd3` | download: `hf_transfer` + only fetch `process_checkpoint/*` (skip the 28 GB DiT) |
| `2e7e802` | `uv venv --clear` (old venv caused an interactive "replace?" hang); **diffusers from git main** (`WanAnimatePipeline` not in a stable release yet); **`mode="replace"`** on the inference call (default is `"animate"` — would ignore bg/mask = wrong output) |
| `adc43d6` | in-process `snapshot_download` instead of `huggingface-cli` subprocess; strip `-e` from the SAM2 git req (uv rejects editable git sources → SAM2 was silently not installing, which aborted the WHOLE animate-reqs install) |
| `3808421` | add `moviepy` + `imageio-ffmpeg` to venv |

## Known gotchas / facts
- `preprocess_data.py --replace_flag` reads only `process_checkpoint/{det,pose2d,sam2}` — the full 28 GB model is NOT needed for preprocessing (inference uses the separate `-Diffusers` repo).
- Wan2.2 `requirements.txt` pins `flash_attn` (compile-heavy) but preprocessing never imports it — we strip it.
- `WanAnimatePipeline.__call__` signature verified against diffusers `main`: args used (`image, pose_video, face_video, background_video, mask_video, prompt, negative_prompt, height, width, segment_frame_length, prev_segment_conditioning_frames, guidance_scale, generator`) are correct; **`mode="replace"` is required** for replacement.
- Replace mode's built-in mask extractor is **single-person only** — use a single-subject source clip (or bring your own mask; the 04b VACE route handles multi-subject via SAM2).

## Improvements to consider (from a best-practices audit)
- Enable CFG (`guidance_scale>1.0`, e.g. 5.0) for stronger face/prompt control (§8C) — currently 1.0.
- Relighting LoRA (§8B) for better scene integration.
- LightX2V 4-step distillation (§8D) for much faster iteration.
- Consider caching `process_checkpoint` to Drive to avoid re-downloading ~4 GB each session.

## Operating playbook (READ THIS — Colab specifics)
**You (the agent) edit code, but you cannot do everything in the browser. Be proactive; only
ask the user for the few things below.**

1. **Changing notebook code:** the notebook is opened *directly from GitHub*. To change it: edit the `.ipynb` locally, `git commit` + `git push origin main`, then the Colab page must be **reloaded** (Cmd+R) to pull it. Editing cells directly in the browser via automation is unreliable (CodeMirror auto-indent mangles multi-line Python).
2. **The reload needs the user.** Colab fires a "Leave site?" (beforeunload) dialog that browser automation cannot dismiss (tried: force-navigate, JS `location.reload()`, neutering `returnValue`/`preventDefault` — all swallowed). **Ask the user to hard-reload the tab.**
3. **`files.upload()` needs the user.** It renders in a sandboxed cross-origin iframe; automation cannot drive the native file picker. **Ask the user to upload** `walking-down-street.mp4` + `reference.png` (both in `~/Downloads`). Note: files.upload appends `(1)` to names if a prior file exists — read the printed `SOURCE_VIDEO =`/`CHARACTER_REF =` values, don't assume names.
4. **After a reload, do NOT "Run all"** — cell 2 would re-prompt for upload. Run cells individually from cell 3 onward; a page reload keeps the runtime, so kernel vars/uploads/Drive persist. A **full session restart** (or "add new GPU") wipes `/content` + vars → then you DO need cells 1–2 again (and a re-upload).
5. **"This notebook was not authored by Google"** warning appears on first cell run per session → click **Run anyway**.
6. **Reading errors:** the user can paste error text — that's faster than screenshotting. `subprocess.run(..., check=True)` without capture hides stderr; prefer in-process calls or read the log file the cell writes.
7. **GPU:** A100 High-RAM. Inference ~28 GB bf16 → A100-40 uses group offloading (already coded), A100-80 runs resident.

**Ask the user to step in for:** hard-reloading a tab after a push · uploading input files · changing runtime type / adding a GPU · any Google-auth popup (Drive mount). Everything else (diagnosing errors, editing+pushing fixes, running cells, checking outputs) you do yourself.

**Definition of done:** cell 6 produces `{ts}_replace.mp4` in `VID_OUT` (Drive) and the displayed video shows the source person replaced by the Yuna character with the original background/motion preserved.
