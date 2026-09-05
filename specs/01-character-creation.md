# Spec 01 — Character Creation (Training)

## Goal
Given N reference images of an original fictional character, produce an SDXL character LoRA
that can be triggered by a unique token to generate consistent stills of that character.

## Inputs
| Input | Detail |
|---|---|
| Character name | e.g. "Aria" |
| Reference images | 10–30 JPG/PNG; mixed poses, angles, expressions, lighting, backgrounds |
| (optional) Trigger token | auto-generated as `ohwx_<name>` if not supplied |
| (optional) Character notes | backstory/appearance notes for captioning guidance |

## Step 1 — Auto-caption reference images

Tool: **JoyCaption** (locally) or **WD14-tagger** (simpler, faster on Colab).

For each image:
1. Run JoyCaption to get a descriptive caption.
2. Prepend the trigger token: `"ohwx_aria, <caption>"`.
3. Save as `characters/<name>/captions/<filename>.txt` (sidecar to the image).

JoyCaption Colab: `prototype/01a_caption_refs.ipynb`
Alternative: `engine/training/captioning/caption_refs.py` (local, uses Transformers).

**What to caption:** describe what *varies* (pose, expression, clothing, setting, background).
The trigger token carries the invariant identity — do not describe the face/body as if 
defining the character; let the LoRA learn that from the image.

## Step 2 — ai-toolkit SDXL LoRA training

Config template: `engine/training/configs/sdxl_character_lora.yaml`

Key hyperparameters:
```yaml
model:
  name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"  # or local path
  is_v2: false

train:
  batch_size: 1
  steps: 2000               # 1500–3000; start at 2000, eval sample every 250 steps
  lr: 1.0e-4
  optimizer: adamw8bit
  gradient_checkpointing: true

network:
  type: "lora"
  linear: 16               # rank; bump to 32 for more identity detail
  linear_alpha: 8

sample:
  sampler: euler
  sample_every: 250
  width: 1024
  height: 1024
  prompts:
    - "ohwx_aria, portrait, detailed face"
    - "ohwx_aria, full body shot, standing"
```

VRAM requirements: ~10–12 GB (rank 16) to ~14–16 GB (rank 32) on a single GPU.
Fits a local 24 GB card. On Colab: use A100 (40 GB) for headroom.
Training time: ~20–40 min at 2000 steps on RTX 4090 / A100.

Colab notebook: `prototype/01b_train_sdxl_lora.ipynb`
Local script: `engine/training/train_sdxl.sh`

## Step 3 — Register in character library

After training:
```python
from library.library import CharacterLibrary
lib = CharacterLibrary()
char = lib.create_character("Aria", trigger="ohwx_aria")
lib.set_lora(char["id"], "characters/Aria/loras/aria_sdxl.safetensors")
lib.add_ref(char["id"], "characters/Aria/reference-images/ref01.jpg", caption="...")
```

Copy the LoRA into `characters/<name>/loras/` and update `characters/<name>/metadata.json`.

## Step 4 (optional) — Wan 2.2 video LoRA

If you intend to use this character heavily in Modes 1 and 3, also train a Wan 2.2 video LoRA
for identity retention through motion.

Tool: **musubi-tuner** (`engine/training/train_wan_video.sh`)
Dataset: the same reference images (image-only dataset is sufficient; video clips add motion fidelity)
VRAM: ~12 GB (image dataset) / ~24 GB (video dataset) with FP8 + block-swap
Time: ~30–90 min

Register: `lib.set_video_lora(char["id"], "characters/Aria/loras/aria_wan22.safetensors")`

## Outputs
- `characters/<name>/loras/<name>_sdxl.safetensors` — the character LoRA
- `characters/<name>/loras/<name>_wan22.safetensors` — optional video LoRA
- `characters/<name>/metadata.json` — updated
- `library/characters.db` — updated row in `characters` table
- Sample images at `characters/<name>/samples/` (produced by ai-toolkit during training)

## Validation
Generate 4 test images with the trigger token using Spec 02 stills pipeline.
Eyeball identity consistency vs reference images. If face drifts badly: increase rank to 32 and
train 500 more steps. If overfitting (background bleeds in): reduce steps by 250.
