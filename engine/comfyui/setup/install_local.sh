#!/usr/bin/env bash
# install_local.sh — Local setup for ComfyUI + all required custom nodes + model downloads.
# Tested on macOS (CPU only) and Linux with CUDA 12.x (GPU).
# Run from the project root: bash engine/comfyui/setup/install_local.sh
set -euo pipefail

# ─── config ──────────────────────────────────────────────────────────────────
COMFYUI_DIR="${COMFYUI_DIR:-$HOME/ComfyUI}"
VENV_DIR="${COMFYUI_DIR}/.venv"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
CUDA_VERSION="${CUDA_VERSION:-cu121}"   # cu121, cu124, cpu
HF_TOKEN="${HF_TOKEN:-}"                # set for gated models (SDXL, etc.)
SKIP_MODELS="${SKIP_MODELS:-0}"         # set to 1 to skip model downloads

echo "=== AI Character Studio — Local Setup ==="
echo "ComfyUI dir: $COMFYUI_DIR"
echo "Torch: $TORCH_VERSION+$CUDA_VERSION"
echo ""

# ─── 1. Clone / update ComfyUI ──────────────────────────────────────────────
if [ ! -d "$COMFYUI_DIR" ]; then
  echo "[1/6] Cloning ComfyUI..."
  git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
else
  echo "[1/6] Updating ComfyUI..."
  git -C "$COMFYUI_DIR" pull --ff-only
fi

# ─── 2. Python venv + PyTorch ───────────────────────────────────────────────
echo "[2/6] Setting up Python venv..."
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [ "$CUDA_VERSION" = "cpu" ]; then
  pip install torch=="${TORCH_VERSION}" torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch=="${TORCH_VERSION}" torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
fi
pip install -r "$COMFYUI_DIR/requirements.txt"
pip install huggingface_hub diffusers transformers accelerate

# ─── 3. Custom nodes ────────────────────────────────────────────────────────
echo "[3/6] Installing custom nodes..."
NODES_DIR="$COMFYUI_DIR/custom_nodes"
mkdir -p "$NODES_DIR"

install_node() {
  local repo="$1"
  local name
  name=$(basename "$repo" .git)
  if [ ! -d "$NODES_DIR/$name" ]; then
    echo "  → $name"
    git clone --depth 1 "$repo" "$NODES_DIR/$name"
    if [ -f "$NODES_DIR/$name/requirements.txt" ]; then
      pip install -r "$NODES_DIR/$name/requirements.txt" || true
    fi
  else
    echo "  ✓ $name (exists)"
  fi
}

# Core utility
install_node https://github.com/ltdrussell/ComfyUI-VideoHelperSuite
install_node https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite   # VHS — pick one
install_node https://github.com/cubiq/ComfyUI_IPAdapter_plus
install_node https://github.com/Fannovel16/comfyui_controlnet_aux
install_node https://github.com/ltdrussell/ComfyUI-Advanced-ControlNet
install_node https://github.com/Acly/comfyui-tooling-nodes

# Video models
install_node https://github.com/Lightricks/ComfyUI-LTXVideo          # LTX-Video (official)
install_node https://github.com/kijai/ComfyUI-WanVideoWrapper         # Wan VACE, Animate, FLF2V
install_node https://github.com/kijai/ComfyUI-LTXVideo                # alt LTX nodes (Kijai)

# Upscale
install_node https://github.com/ByteDance-Seed/ComfyUI-SeedVR2        # check for official node

# SAM 2
install_node https://github.com/kijai/ComfyUI-segment-anything-2

# Impact pack (face detection, etc.)
install_node https://github.com/ltAstrid/ComfyUI-Impact-Pack
install_node https://github.com/pythongosssss/ComfyUI-Custom-Scripts

# PuLID for FLUX (optional, install if testing FLUX later)
# install_node https://github.com/cubiq/PuLID_ComfyUI

echo "[3/6] Custom nodes done."

# ─── 4. Model downloads ─────────────────────────────────────────────────────
if [ "$SKIP_MODELS" = "1" ]; then
  echo "[4/6] Skipping model downloads (SKIP_MODELS=1)."
else
  echo "[4/6] Downloading models..."
  MODELS_DIR="$COMFYUI_DIR/models"

  hf_download() {
    local repo="$1" file="$2" dest="$3"
    mkdir -p "$(dirname "$MODELS_DIR/$dest")"
    if [ ! -f "$MODELS_DIR/$dest" ]; then
      echo "  → $file"
      python3 -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id='$repo', filename='$file',
    local_dir=os.path.dirname('$MODELS_DIR/$dest'),
    token='${HF_TOKEN}' or None,
)
"
    else
      echo "  ✓ $file (exists)"
    fi
  }

  # SDXL base
  hf_download "stabilityai/stable-diffusion-xl-base-1.0" \
    "sd_xl_base_1.0.safetensors" "checkpoints/sdxl_base_1.0.safetensors"

  # SDXL VAE (fp16 fixed)
  hf_download "madebyollin/sdxl-vae-fp16-fix" \
    "sdxl_vae.safetensors" "vae/sdxl_vae_fp16fix.safetensors"

  # ControlNet — SDXL depth
  hf_download "diffusers/controlnet-depth-sdxl-1.0" \
    "diffusion_pytorch_model.fp16.safetensors" "controlnet/sdxl_depth.safetensors"

  # ControlNet — SDXL OpenPose
  hf_download "thibaud/controlnet-openpose-sdxl-1.0" \
    "OpenPoseXL2.safetensors" "controlnet/sdxl_openpose.safetensors"

  # IP-Adapter FaceID Plus v2 SDXL
  hf_download "h94/IP-Adapter-FaceID" \
    "ip-adapter-faceid-plusv2_sdxl.bin" "ipadapter/ip-adapter-faceid-plusv2_sdxl.bin"
  # LoRA sidecar — required alongside the adapter weights above
  hf_download "h94/IP-Adapter-FaceID" \
    "ip-adapter-faceid-plusv2_sdxl_lora.safetensors" "ipadapter/ip-adapter-faceid-plusv2_sdxl_lora.safetensors"

  # LTX-Video — verified filenames Sept 2026 (naming changed in v0.9.6+)
  # 2B distilled FP8 — 4.5 GB, fastest, fits local 24 GB, multi-keyframe
  hf_download "Lightricks/LTX-Video" \
    "ltxv-2b-0.9.8-distilled-fp8.safetensors" "video/ltxv-2b-0.9.8-distilled-fp8.safetensors"
  # Spatial upscaler — 505 MB, worth having
  hf_download "Lightricks/LTX-Video" \
    "ltxv-spatial-upscaler-0.9.8.safetensors" "video/ltxv-spatial-upscaler-0.9.8.safetensors"
  # 13B dev FP8 — 15.7 GB, A100 needed, best quality (uncomment to download)
  # hf_download "Lightricks/LTX-Video" "ltxv-13b-0.9.8-dev-fp8.safetensors" "video/ltxv-13b-0.9.8-dev-fp8.safetensors"

  # Wan VACE 1.3B — sharded diffusers format, must use snapshot_download
  echo "  -> Wan2.1-VACE-1.3B (snapshot, ~7 GB)..."
  python3 -c "
from huggingface_hub import snapshot_download, constants
import os
dest = '${MODELS_DIR}/video/Wan2.1-VACE-1.3B'
if not os.path.isdir(dest) or not os.listdir(dest):
    snapshot_download('Wan-AI/Wan2.1-VACE-1.3B', local_dir=dest,
                      ignore_patterns=['*.msgpack','*.h5','flax_model*'])
print('  Wan2.1-VACE-1.3B ready at', dest)
"

  echo "[4/6] Model downloads done."
  echo "NOTE: Wan2.1-FLF2V-14B, Wan2.2-Animate-14B need 14–24GB+ — download on A100 Colab."
fi

# ─── 5. Symlink project models dir ─────────────────────────────────────────
echo "[5/6] Linking project models dir..."
PROJECT_MODELS="$(pwd)/models"
mkdir -p "$PROJECT_MODELS"
# ComfyUI extra_model_paths.yaml
cat > "$COMFYUI_DIR/extra_model_paths.yaml" <<EOF
character_studio:
  base_path: ${PROJECT_MODELS}
  checkpoints: checkpoints/
  loras: loras/
  vae: vae/
  controlnet: controlnet/
  ipadapter: ipadapter/
  video: video/
  upscale: upscale/
EOF
echo "  extra_model_paths.yaml written."

# ─── 6. Launch script ───────────────────────────────────────────────────────
cat > run_comfyui.sh <<'EOF'
#!/usr/bin/env bash
source "$HOME/ComfyUI/.venv/bin/activate"
cd "$HOME/ComfyUI"
python main.py --listen 0.0.0.0 --port 8188 --preview-method auto "$@"
EOF
chmod +x run_comfyui.sh

echo ""
echo "=== Setup complete ==="
echo "Start ComfyUI:   bash run_comfyui.sh"
echo "Open UI:         http://localhost:8188"
echo "Install project deps: pip install -r requirements.txt"
echo ""
echo "Next: open Colab notebooks in prototype/ for heavy training/inference."
