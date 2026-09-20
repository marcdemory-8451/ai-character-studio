# Handoff — 04b: Wan VACE 1.3B "swap-anything" (low-VRAM alternate)

**Goal:** Take a source video + a character reference still, use **SAM2** to track/mask the
source subject, then **Wan VACE** regenerates the masked region as your character (keeping the
background). Lighter than 04a (~8–10 GB at 480p), handles multi-subject via explicit masks.
Pure-`diffusers` `WanVACEPipeline` on Colab.

## Links
- **Repo:** https://github.com/marcdemory-8451/ai-character-studio (branch `main`)
- **Local clone:** `/Users/m626288/CLAUDE_PROJECTS/ai-character-studio`
- **Colab (open from GitHub):** https://colab.research.google.com/github/marcdemory-8451/ai-character-studio/blob/main/prototype/04b_test_video_mode2_vace.ipynb
- **Notebook file:** `prototype/04b_test_video_mode2_vace.ipynb`
- **Models:** `Wan-AI/Wan2.1-VACE-1.3B-diffusers` (inference) · SAM2: repo `facebookresearch/sam2` (code) + `facebook/sam2.1-hiera-large` (checkpoint `sam2.1_hiera_large.pt`)

## Architecture
1. **Config + mount Drive** — `CHARACTER_NAME='Yuna'`, `WIDTH,HEIGHT=832,480`, `MAX_FRAMES=81`.
2. **Upload inputs** — `files.upload()` for SOURCE VIDEO + CHARACTER REFERENCE still.
3. **Install** — diffusers stack (system env); clone `facebookresearch/sam2` → `/content/sam2_repo`, `pip install -e`; download SAM2.1 checkpoint; import-check.
4. **SAM2 tracking → mask** — one foreground point `PT=(0.5,0.4)` on frame 0, propagate across frames → per-frame binary mask (white = regenerate). Shows a mask montage to review.
5. **Load `WanVACEPipeline` + run** — `video=source frames`, `mask=mask frames`, `reference_images=[char]` → swapped clip. VAE fp32, transformer bf16, resident ≥30 GB else group offload.
6. Metadata log. 7. Commented alternates (14B @ 720p, DWPose control, first/last-frame, conditioning_scale knob).

## Current status (2026-09-20, session 2)

**Working (verified this session):**
- Cells 1 (config+mount) ✅, 2 (upload `walking-down-street.mp4` + reference) ✅.
- Cell 3 install: pip/uv installs, SAM2 editable build, and the 898 MB SAM2.1 checkpoint
  download all succeed. The ONLY remaining failure was the final `sam2 OK` verify import
  (the shadow guard) — **now fixed by `515ed70`, awaiting a re-run to confirm.**

**Fixed this session (all on `main`, pushed — the notebook opened from GitHub `main`):**
- Cell 4 (SAM2 tracking) had 4 latent API-misuse crashes (never reached before — earlier
  runs died at the checkpoint/import bugs). Rewrote against SAM2 + diffusers source.
- The SAM2 import-shadow guard was still firing even with the `sam2_repo` clone name.

**⏭️ IMMEDIATE NEXT STEP (pick up here):**
1. **Hard-reload the Colab tab** (Cmd+R → "Leave") to pull `515ed70`. Runtime + uploads persist.
2. Re-run **cell 3** → must end with `torch … | diffusers … | sam2 OK` (this was the failing line).
3. Run **cell 4** (SAM2 mask) → expect frame count, `tracked 0/…`, and the **mask montage**.
   **Review the montage**: the white region must cover the walking person. If it misses/drifts,
   change `PT = (x, y)` (normalized 0–1) in the cell to sit on the subject in frame 0, re-run.
4. Run **cell 5** (`code5` load + `code6` swap) — **NEVER RUN YET; the real unknown.** Watch for:
   - `from diffusers import WanVACEPipeline` / `AutoencoderKLWan` resolving on the pinned
     `diffusers>=0.35.0`. If `ImportError`, install diffusers from git main (like 04a) and reload.
   - VRAM strategy: A100-80GB → resident; smaller → group offload (both coded in `code5`).
   - Output `{ts}_swap.mp4` in `VID_OUT` on Drive. Then cell 6 logs metadata.
5. If quality is off: `conditioning_scale` 0.5–0.8 (§7D), or step up to VACE-14B @ 720p (§7A).

**Cell 5 (VACE call) was audited against diffusers `WanVACEPipeline` source and is believed
correct** (`video`/`mask`/`reference_images`/`conditioning_scale`; white=regenerate matches) —
no code change made, but it is unvalidated on GPU.

## Fixes already applied (all on `main`)
| Commit | Fix |
|---|---|
| `adea5c9` | clone SAM2 to `/content/sam2_repo`; use Hydra config **name** `configs/sam2.1/sam2.1_hiera_l.yaml` |
| `905a31a` | SAM2 checkpoint repo id → `facebook/sam2.1-hiera-large` (was `facebook/sam2`, a 404) |
| `fad86cc` | Cell 4 rewrite — correct SAM2 API: init_state needs a **JPEG-frame dir** (not numpy) → writes `/content/sam2_frames/<i>.jpg`; `add_new_points_or_box` needs `obj_id=1` + np arrays; `propagate_in_video` is a **generator over all frames**; masks are **logits** → threshold `>0`; trim to Wan-legal `4k+1` |
| `f744a27` | (superseded) tried `chdir` into repo before import — a **no-op in Colab** (sys.path holds the literal `/content`, not `''`) |
| `515ed70` | SAM2 shadow guard fixed properly: editable install makes `sam2` a **namespace pkg** whose `__path__[0]` is the repo root; `build_sam.py` raises because `repo_root/sam2` exists. Fix: `import sam2` (guard-free), prune `__path__` to entries whose basename==`sam2`, then import `build_sam`. In cells 3 + 4. |

## Known gotchas / facts
- **SAM2 import-shadow guard (the real story):** `build_sam.py` raises `RuntimeError("running from the parent directory of the sam2 repo")` when `os.path.isdir(os.path.join(sam2.__path__[0], "sam2"))`. The `pip install -e` editable install makes `sam2` a **namespace package** whose `__path__` includes the repo ROOT (`/content/sam2_repo`) as `__path__[0]`, and `repo_root/sam2` exists → guard fires. `chdir` does NOT help in Colab (sys.path holds the literal `/content`, not the dynamic `''`). The working fix (`515ed70`): `import sam2` (its `__init__` is guard-free, only inits hydra), then `sam2.__path__ = [p for p in sam2.__path__ if os.path.basename(p)=='sam2']`, THEN `from sam2.build_sam import ...`.
- **SAM2 config:** `build_sam2_video_predictor` wants a Hydra config *name* resolved via the `sam2` package's registered search path, not an absolute path.
- **SAM2 checkpoint/config pairing:** 2.1 checkpoint (`sam2.1_hiera_large.pt`) ↔ 2.1 config (`sam2.1_hiera_l.yaml`) — already matched correctly.
- **VACE mask convention:** BLACK = preserve/condition, WHITE = regenerate.
- `WanVACEPipeline` is available in diffusers ≥0.35.0 (04b installs `diffusers>=0.35.0`, not git — verify at cell 5 that `from diffusers import WanVACEPipeline` and `model_id="Wan-AI/Wan2.1-VACE-1.3B-diffusers"` both resolve; if not, install diffusers from git main like 04a does).
- 1.3B @ 480p ≈ 8–10 GB → fits smaller GPUs; A100 runs it resident.

## Improvements to consider (from a best-practices audit)
- fp8/int8 transformer quantization + a distilled/CausVid-style acceleration LoRA → more frames per session.
- 14B @ 720p (§7A) for higher fidelity once 1.3B is validated.
- `conditioning_scale` knob (§7D, 0.5–0.8) if background bleeds or source pose dominates.

## Operating playbook (READ THIS — Colab specifics)
**You (the agent) edit code, but you cannot do everything in the browser. Be proactive; only
ask the user for the few things below.**

1. **Changing notebook code:** opened directly from GitHub. Edit the `.ipynb` locally, `git commit` + `git push origin main`, then the Colab page must be **reloaded** (Cmd+R) to pull it. Don't hand-edit cells via automation (auto-indent mangles multi-line Python).
2. **The reload needs the user.** Colab's "Leave site?" (beforeunload) dialog cannot be dismissed by automation. **Ask the user to hard-reload the tab.**
3. **`files.upload()` needs the user** (sandboxed iframe; can't drive the native picker). **Ask the user to upload** `walking-down-street.mp4` + `reference.png` (in `~/Downloads`). Read the printed `SOURCE_VIDEO =`/`CHARACTER_REF =` values (names may get a `(1)` suffix).
4. **After a reload, do NOT "Run all"** (re-triggers upload on cell 2). Run cells individually from cell 3; a page reload keeps the runtime (vars/uploads/Drive persist). A **full session restart / new GPU** wipes `/content` + vars → redo cells 1–2 (re-upload).
5. **"Not authored by Google"** warning on first run per session → **Run anyway**.
6. **Reading errors:** ask the user to paste error text (faster than screenshots).
7. **GPU:** A100 High-RAM is plenty; the 1.3B model is small.

**Ask the user to step in for:** hard-reloading a tab after a push · uploading input files · changing runtime type / adding a GPU · Google-auth popups (Drive mount). Everything else you do yourself.

**Definition of done:** cell 4 shows a clean mask that tracks the subject; cell 5 produces
`{ts}_swap.mp4` in `VID_OUT` (Drive) showing the source subject replaced by the Yuna character
with the background preserved.
