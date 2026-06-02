#!/bin/bash
# Offline SLURM job: run the olmOCR pipeline over MDPBench on a Trillium H100, then evaluate.
# Submit from the login node AFTER the prep stage (venvs built, dataset + model weights cached).
#   sbatch scripts/trillium/mdpbench_job.sbatch
#
#SBATCH --account=def-consens
#SBATCH --job-name=olmocr-mdpbench
#SBATCH --partition=compute
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/rainy/mdpbench/slurm-%j.out

set -euo pipefail

WORK=/scratch/rainy/mdpbench
MODEL=allenai/olmOCR-2-7B-1025-FP8

# Offline: compute nodes have no internet; everything was pre-fetched in the prep stage.
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== node: $(hostname) ==="
nvidia-smi -L || true

# ---------------------------------------------------------------------------
# Stage A: olmOCR pipeline (olmocr venv + opencv module for vllm's cv2)
# ---------------------------------------------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.11.5 opencv/4.12.0
source "$WORK/venv/bin/activate"
cd "$WORK"

echo "=== convert MDPBench images to PDFs ==="
python "$WORK/glue_convert_images.py"

echo "=== run olmOCR pipeline ($MODEL) ==="
rm -rf localworkspace
# shellcheck disable=SC2086  # intentional glob expansion into --pdfs (nargs=*)
python -m olmocr.pipeline ./localworkspace --markdown --pdfs ./mdp_pdfs/*.pdf --model "$MODEL"

echo "=== flatten predictions into MDPBench layout ==="
python "$WORK/glue_flatten_md.py"

deactivate

# ---------------------------------------------------------------------------
# Stage B: MDPBench evaluation (separate eval venv; CDM skipped)
# ---------------------------------------------------------------------------
module load python/3.11.5
source "$WORK/eval_venv/bin/activate"
cd "$WORK/MultimodalOCR/MDPBench"

export MDP_GT_PATH="$WORK/MDPBench_dataset/MDPBench_public.json"
cp "$WORK/glue_patch_config.py" .
python glue_patch_config.py

echo "=== run MDPBench end2end evaluation ==="
python pdf_validation.py --config ./configs/end2end_olmocr.yaml

echo "=== calculate scores ==="
RESULT_FOLDER="result/olmocr_result"
if [ ! -d "$RESULT_FOLDER" ]; then
    RESULT_FOLDER=$(find result -maxdepth 1 -type d -name "*_result" | head -n1)
    echo "Using detected result folder: $RESULT_FOLDER"
fi
python tools/calculate_scores.py --result_folder "$RESULT_FOLDER"

echo "=== DONE: scores under $WORK/MultimodalOCR/MDPBench/$RESULT_FOLDER ==="
