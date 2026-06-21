#!/usr/bin/env bash
# One-shot bootstrap to populate a RunPod Network Volume with LTX-2.3 weights
# + the 12 LoRAs. Run ONCE on a tiny RunPod CPU Pod (or any GPU pod) with the
# Network Volume attached at /workspace.
#
# Usage on the bootstrapper Pod:
#   export HF_TOKEN=hf_xxx
#   bash prepare_volume.sh
#
# Cost: ~15-30 min on a $0.20/hr CPU pod. ≈ $0.10 total.

set -euo pipefail

MODELS=/workspace/models
LORAS=/workspace/loras
mkdir -p "$MODELS" "$LORAS"

# Slow but reliable downloader (hf_transfer stalls on some files).
export HF_HUB_ENABLE_HF_TRANSFER=0
pip install --quiet 'huggingface_hub[cli]'

hf_file() {
    local repo="$1"; local file="$2"; local dest="$3"
    if [ -f "$dest/$file" ]; then
        echo "[skip] $dest/$file"; return
    fi
    echo "[get ] $repo :: $file"
    hf download "$repo" "$file" --local-dir "$dest"
}

hf_repo() {
    local repo="$1"; local dest="$2"; local sentinel="$3"
    if [ -f "$dest/$sentinel" ]; then
        echo "[skip] $dest"; return
    fi
    mkdir -p "$dest"
    echo "[get ] $repo -> $dest"
    hf download "$repo" --local-dir "$dest"
}

# Core LTX-2.3
hf_file "Lightricks/LTX-2.3" "ltx-2.3-22b-distilled-1.1.safetensors"          "$MODELS"
hf_file "Lightricks/LTX-2.3" "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"    "$MODELS"
hf_file "Lightricks/LTX-2.3" "ltx-2.3-22b-distilled-lora-384-1.1.safetensors" "$MODELS"

# Gemma-3 (gated)
hf_repo "google/gemma-3-12b-it-qat-q4_0-unquantized" \
        "$MODELS/gemma-3-12b-it-qat-q4_0-unquantized" \
        "model-00005-of-00005.safetensors"

# LoRAs — same 12 as the Modal app.
LORA_REPOS=(
    "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"
    "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control"
    "Lightricks/LTX-2.3-22b-IC-LoRA-Colorization"
    "Lightricks/LTX-2.3-22b-IC-LoRA-Day-To-Night"
    "Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting"
    "Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients"
    "Lightricks/LTX-2.3-22b-IC-LoRA-Decompression"
    "Lightricks/LTX-2.3-22b-IC-LoRA-Deblur"
    "Lightricks/LTX-2.3-22b-IC-LoRA-Cross-Eyed"
    "Lightricks/LTX-2.3-22b-IC-LoRA-Water-Simulation"
    "Lightricks/LTX-2.3-22b-IC-LoRA-Instant-Shave"
    "Zlikwid/LTX_2.3_Upscale_IC_Lora"
)
SKIPPED=()
for repo in "${LORA_REPOS[@]}"; do
    echo "[lora] $repo"
    if ! hf download "$repo" --include "*.safetensors" --local-dir "$LORAS"; then
        SKIPPED+=("$repo")
    fi
done

echo ""
echo "================================================"
du -sh "$MODELS" "$LORAS"
ls -1 "$LORAS"/*.safetensors | wc -l | xargs echo "LoRAs on disk:"
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "SKIPPED (likely gated — accept license on HF):"
    for repo in "${SKIPPED[@]}"; do
        echo "  https://huggingface.co/$repo"
    done
fi
echo "================================================"
