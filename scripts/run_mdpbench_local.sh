#!/bin/bash

# Runs the olmOCR pipeline against MDPBench (https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/MDPBench)
# DIRECTLY on a GPU machine -- no Beaker, no Docker. Run this from inside an environment that
# already has olmocr installed (pip install -e .[gpu]) and a working GPU.
#
# It will:
#   1. Download the MDPBench dataset (images + ground truth) from HuggingFace
#   2. Convert each benchmark image to a single page PDF
#   3. Run `python -m olmocr.pipeline ... --markdown` to produce per-image markdown
#   4. Flatten the markdown into the MDPBench prediction layout (result/olmocr/<name>.md)
#   5. Run the MDPBench end2end evaluation (incl. the CDM formula metric) and print scores
#
# Usage:
#   ./scripts/run_mdpbench_local.sh                                 # default model + dataset, CDM enabled
#   ./scripts/run_mdpbench_local.sh --model allenai/olmOCR-2-7B-1025-FP8
#   ./scripts/run_mdpbench_local.sh --benchrepo Delores-Lin/MDPBench --benchbranch main
#   ./scripts/run_mdpbench_local.sh --benchpath /data/MDPBench      # use an already-downloaded copy
#   ./scripts/run_mdpbench_local.sh --workdir /scratch/mdpbench     # where to put data + outputs
#   ./scripts/run_mdpbench_local.sh --skip-cdm                      # skip heavy CDM setup (Edit_dist still scores formulas)
#
# Requirements: olmocr (with GPU deps) on PATH, git, uv (auto-installed if missing).
# Full CDM additionally needs sudo for: texlive-full, ImageMagick build deps, Node 16.

set -euo pipefail

MODEL="allenai/olmOCR-2-7B-1025-FP8"
BENCH_REPO="Delores-Lin/MDPBench"
BENCH_BRANCH=""
BENCH_PATH=""
WORKDIR="$(pwd)/mdpbench_run"
SKIP_CDM="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)      MODEL="$2"; shift 2 ;;
        --benchrepo)  BENCH_REPO="$2"; shift 2 ;;
        --benchbranch) BENCH_BRANCH="$2"; shift 2 ;;
        --benchpath)  BENCH_PATH="$2"; shift 2 ;;
        --workdir)    WORKDIR="$2"; shift 2 ;;
        --skip-cdm)   SKIP_CDM="1"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Model:     $MODEL"
echo "Workdir:   $WORKDIR"
echo "CDM:       $([ "$SKIP_CDM" = "1" ] && echo disabled || echo enabled)"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Ensure uv is available (used for the isolated eval venv)
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    pip install uv
fi

# ---------------------------------------------------------------------------
# 1. Download the MDPBench dataset
# ---------------------------------------------------------------------------
if [ -n "$BENCH_PATH" ]; then
    echo "Using existing benchmark data at $BENCH_PATH"
    rm -rf MDPBench_dataset
    cp -r "$BENCH_PATH" MDPBench_dataset
else
    echo "Downloading MDPBench dataset from HuggingFace ($BENCH_REPO)..."
    HF_DL=(hf download --repo-type dataset "$BENCH_REPO" --max-workers 4 --local-dir ./MDPBench_dataset)
    if [ -n "$BENCH_BRANCH" ]; then
        HF_DL+=(--revision "$BENCH_BRANCH")
    fi
    "${HF_DL[@]}"
fi

if [ ! -f MDPBench_dataset/MDPBench_public.json ]; then
    echo "ERROR: MDPBench_dataset/MDPBench_public.json not found after download." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Convert MDPBench images to single page PDFs (basename preserved)
# ---------------------------------------------------------------------------
cat > /tmp/mdp_convert_images.py << 'PYEOF'
import glob, os, sys
from olmocr.image_utils import convert_image_to_pdf_bytes

img_dir = "MDPBench_dataset/MDPBench_img_public"
out_dir = "mdp_pdfs"
os.makedirs(out_dir, exist_ok=True)

exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG", "*.tif", "*.tiff", "*.webp", "*.bmp")
images = []
for ext in exts:
    images.extend(glob.glob(os.path.join(img_dir, "**", ext), recursive=True))
images = sorted(set(images))
print(f"Found {len(images)} MDPBench images")

seen = {}
written = 0
for img_path in images:
    base = os.path.splitext(os.path.basename(img_path))[0]
    if base in seen:
        print(f"WARNING: duplicate image basename '{base}' ({img_path} vs {seen[base]}); skipping", file=sys.stderr)
        continue
    seen[base] = img_path
    try:
        pdf_bytes = convert_image_to_pdf_bytes(img_path)
    except Exception as e:
        print(f"WARNING: failed to convert {img_path}: {e}", file=sys.stderr)
        continue
    with open(os.path.join(out_dir, base + ".pdf"), "wb") as f:
        f.write(pdf_bytes)
    written += 1

print(f"Wrote {written} PDFs to {out_dir}")
PYEOF

echo "Converting images to PDFs..."
python /tmp/mdp_convert_images.py

# ---------------------------------------------------------------------------
# 3. Run the olmocr pipeline (manages its own vllm server)
# ---------------------------------------------------------------------------
echo "Running olmocr pipeline with model $MODEL..."
rm -rf localworkspace
python -m olmocr.pipeline ./localworkspace --markdown --pdfs "./mdp_pdfs/*.pdf" --model "$MODEL"

# ---------------------------------------------------------------------------
# 4. Clone MDPBench eval code and flatten predictions to result/olmocr/<name>.md
# ---------------------------------------------------------------------------
if [ ! -d MultimodalOCR ]; then
    echo "Cloning MultimodalOCR (MDPBench eval)..."
    git clone --depth 1 https://github.com/Yuliang-Liu/MultimodalOCR.git
fi

cat > /tmp/mdp_flatten_md.py << 'PYEOF'
import glob, os, shutil, sys

src_glob = "localworkspace/markdown/**/*.md"
dst_dir = "MultimodalOCR/MDPBench/result/olmocr"
os.makedirs(dst_dir, exist_ok=True)

md_files = sorted(glob.glob(src_glob, recursive=True))
print(f"Found {len(md_files)} markdown outputs from the pipeline")

seen = {}
copied = 0
for md in md_files:
    name = os.path.basename(md)
    if name in seen:
        print(f"WARNING: duplicate markdown basename '{name}' ({md} vs {seen[name]}); skipping", file=sys.stderr)
        continue
    seen[name] = md
    shutil.copyfile(md, os.path.join(dst_dir, name))
    copied += 1

print(f"Copied {copied} markdown predictions to {dst_dir}")
PYEOF

echo "Collecting markdown predictions..."
python /tmp/mdp_flatten_md.py

# ---------------------------------------------------------------------------
# 5. Set up the MDPBench eval environment (isolated venv)
# ---------------------------------------------------------------------------
GT_PATH="$(pwd)/MDPBench_dataset/MDPBench_public.json"
cd MultimodalOCR/MDPBench

echo "Creating isolated MDPBench eval venv (python 3.10)..."
uv venv --python 3.10 .mdpbench_venv
# shellcheck disable=SC1091
source .mdpbench_venv/bin/activate
uv pip install -r requirements.txt
uv pip install pyyaml

# ---------------------------------------------------------------------------
# 6. (Optional) CDM formula-metric dependencies: Node 16 + ImageMagick + LaTeX
# ---------------------------------------------------------------------------
if [ "$SKIP_CDM" = "0" ]; then
    echo "Setting up CDM dependencies (needs sudo for apt/make install)..."
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

    $SUDO apt-get update
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends \
        texlive-full build-essential wget git pkg-config

    if ! command -v node >/dev/null 2>&1; then
        wget -q https://registry.npmmirror.com/-/binary/node/latest-v16.x/node-v16.13.1-linux-x64.tar.gz
        tar -xf node-v16.13.1-linux-x64.tar.gz
        $SUDO mkdir -p /usr/local/nodejs
        $SUDO cp -r node-v16.13.1-linux-x64/* /usr/local/nodejs/
        $SUDO ln -sf /usr/local/nodejs/bin/node /usr/local/bin/node
        $SUDO ln -sf /usr/local/nodejs/bin/npm /usr/local/bin/npm
    fi
    node --version

    if ! command -v magick >/dev/null 2>&1; then
        git clone --depth 1 https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.1
        ( cd ImageMagick-7.1.1 && ./configure && make -j"$(nproc)" && $SUDO make install && $SUDO ldconfig /usr/local/lib )
    fi

    if [ -f metrics/cdm/requirements.txt ]; then
        uv pip install -r metrics/cdm/requirements.txt
    fi
fi

# ---------------------------------------------------------------------------
# 7. Patch the eval config and run the evaluation
# ---------------------------------------------------------------------------
cat > /tmp/mdp_patch_config.py << PYEOF
import yaml
gt_path = "$GT_PATH"
skip_cdm = "$SKIP_CDM" == "1"
with open("configs/end2end.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["end2end_eval"]["dataset"]["ground_truth"]["data_path"] = gt_path
cfg["end2end_eval"]["dataset"]["prediction"]["data_path"] = "result/olmocr"
if skip_cdm:
    m = cfg["end2end_eval"]["metrics"]["display_formula"]["metric"]
    cfg["end2end_eval"]["metrics"]["display_formula"]["metric"] = [x for x in m if x != "CDM"]
with open("configs/end2end_olmocr.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print("Wrote configs/end2end_olmocr.yaml")
print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PYEOF

echo "Patching eval config..."
python /tmp/mdp_patch_config.py

echo "Running MDPBench end2end evaluation..."
python pdf_validation.py --config ./configs/end2end_olmocr.yaml

echo "Calculating scores..."
RESULT_FOLDER="result/olmocr_result"
if [ ! -d "$RESULT_FOLDER" ]; then
    RESULT_FOLDER=$(find result -maxdepth 1 -type d -name "*_result" | head -n1)
    echo "Using detected result folder: $RESULT_FOLDER"
fi
python tools/calculate_scores.py --result_folder "$RESULT_FOLDER"

echo ""
echo "Done. Scores are under: $(pwd)/$RESULT_FOLDER"
