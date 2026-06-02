#!/bin/bash

# Runs the olmOCR pipeline against MDPBench (https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/MDPBench),
# a multilingual document parsing benchmark. This submits a Beaker job that:
#   1. Downloads the MDPBench dataset (images + ground truth) from HuggingFace
#   2. Converts each benchmark image to a single page PDF
#   3. Runs `python -m olmocr.pipeline ... --markdown` to produce per-image markdown
#   4. Flattens the markdown into the MDPBench prediction layout (result/olmocr/<name>.md)
#   5. Runs the MDPBench end2end evaluation (including the CDM formula metric) and prints scores
#
# Usage:
#   ./scripts/run_olmocr_mdpbench.sh                                        # Default model + default MDPBench repo
#   ./scripts/run_olmocr_mdpbench.sh --model allenai/olmOCR-2-7B-1025-FP8   # Use a specific olmocr model
#   ./scripts/run_olmocr_mdpbench.sh --benchrepo Delores-Lin/MDPBench       # Use a different benchmark repo
#   ./scripts/run_olmocr_mdpbench.sh --benchbranch main                     # Use a specific dataset branch/revision
#   ./scripts/run_olmocr_mdpbench.sh --benchpath s3://ai2-oe-data/path/     # Use benchmark from S3 or local path
#   ./scripts/run_olmocr_mdpbench.sh --cluster ai2/titan-cirrascale         # Specify a cluster
#   ./scripts/run_olmocr_mdpbench.sh --beaker-image jakep/olmocr-mdpbench-...# Skip Docker build, use existing image

set -e

# Parse command line arguments
MODEL=""
BENCH_BRANCH=""
BENCH_REPO=""
BENCH_PATH=""
CLUSTER=""
BEAKER_IMAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --benchbranch)
            BENCH_BRANCH="$2"
            shift 2
            ;;
        --benchrepo)
            BENCH_REPO="$2"
            shift 2
            ;;
        --benchpath)
            BENCH_PATH="$2"
            shift 2
            ;;
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --beaker-image)
            BEAKER_IMAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL] [--benchrepo REPO] [--benchbranch BRANCH] [--benchpath PATH] [--cluster CLUSTER] [--beaker-image IMAGE]"
            exit 1
            ;;
    esac
done

# Set default olmocr model if not specified
if [ -z "$MODEL" ]; then
    MODEL="allenai/olmOCR-2-7B-1025-FP8"
fi
echo "Using olmocr model: $MODEL"

# Check for mutual exclusivity between benchpath and benchrepo/benchbranch
if [ -n "$BENCH_PATH" ] && ([ -n "$BENCH_REPO" ] || [ -n "$BENCH_BRANCH" ]); then
    echo "Error: --benchpath is mutually exclusive with --benchrepo and --benchbranch"
    echo "Use either --benchpath OR --benchrepo/--benchbranch, not both."
    exit 1
fi

# Use conda environment Python if available, otherwise use system Python
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
    echo "Using conda Python from: $CONDA_PREFIX"
else
    PYTHON="python"
    echo "Warning: No conda environment detected, using system Python"
fi

# Get version from version.py
VERSION=$($PYTHON -c 'import olmocr.version; print(olmocr.version.VERSION)')
echo "OlmOCR version: $VERSION"

# Get first 10 characters of git hash
GIT_HASH=$(git rev-parse HEAD | cut -c1-10)
echo "Git hash: $GIT_HASH"

# Get current git branch name
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Git branch: $GIT_BRANCH"

# Check if a Beaker image was provided
if [ -n "$BEAKER_IMAGE" ]; then
    echo "Using provided Beaker image: $BEAKER_IMAGE"
    IMAGE_TAG="$BEAKER_IMAGE"
else
    # Create full image tag
    IMAGE_TAG="olmocr-benchmark-${VERSION}-${GIT_HASH}"
    echo "Building Docker image with tag: $IMAGE_TAG"

    # Build the Docker image
    echo "Building Docker image..."
    docker build --platform linux/amd64 -f ./Dockerfile -t $IMAGE_TAG .

    # Push image to beaker
    echo "Trying to push image to Beaker..."
    if ! beaker image create --workspace ai2/oe-data-pdf --name $IMAGE_TAG $IMAGE_TAG 2>/dev/null; then
        echo "Warning: Beaker image with tag $IMAGE_TAG already exists. Using existing image."
    fi
fi

# Get Beaker username
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
echo "Beaker user: $BEAKER_USER"

# Create Python script to run beaker experiment
cat << 'EOF' > /tmp/run_mdpbench_experiment.py
import sys
from textwrap import dedent
from beaker import Beaker, BeakerExperimentSpec, BeakerTaskSpec, BeakerTaskContext, BeakerResultSpec, BeakerTaskResources, BeakerImageSource, BeakerJobPriority, BeakerConstraints, BeakerEnvVar

# Get image tag, beaker user, git branch, git hash, and model from command line
image_tag = sys.argv[1]
beaker_user = sys.argv[2]
git_branch = sys.argv[3]
git_hash = sys.argv[4]
model = sys.argv[5]

# Initialize benchmark dataset parameters
bench_branch = None
bench_repo = "Delores-Lin/MDPBench"  # Default repository
bench_path = None
cluster = None

# Parse additional arguments
arg_idx = 6
while arg_idx < len(sys.argv):
    if sys.argv[arg_idx] == "--benchbranch":
        bench_branch = sys.argv[arg_idx + 1]
        arg_idx += 2
    elif sys.argv[arg_idx] == "--benchrepo":
        bench_repo = sys.argv[arg_idx + 1]
        arg_idx += 2
    elif sys.argv[arg_idx] == "--benchpath":
        bench_path = sys.argv[arg_idx + 1]
        arg_idx += 2
    elif sys.argv[arg_idx] == "--cluster":
        cluster = sys.argv[arg_idx + 1]
        arg_idx += 2
    else:
        print(f"Unknown argument: {sys.argv[arg_idx]}")
        arg_idx += 1

# Initialize Beaker client
b = Beaker.from_env(default_workspace="ai2/olmocr")

# Check if AWS credentials secret exists
aws_creds_secret = f"{beaker_user}-AWS_CREDENTIALS_FILE"
try:
    b.secret.get(aws_creds_secret, workspace="ai2/olmocr")
    has_aws_creds = True
    print(f"Found AWS credentials secret: {aws_creds_secret}")
except:
    has_aws_creds = False
    print(f"AWS credentials secret not found: {aws_creds_secret}")

# Check if HF_TOKEN secret exists
hf_token_secret = f"{beaker_user}-HF_TOKEN"
try:
    b.secret.get(hf_token_secret, workspace="ai2/olmocr")
    has_hf_token = True
    print(f"Found HuggingFace token secret: {hf_token_secret}")
except:
    has_hf_token = False
    print(f"HuggingFace token secret not found: {hf_token_secret}")

# Python snippet (run inside the olmocr env) that converts each MDPBench image into a
# single page PDF, preserving the image basename so predictions can be matched by name.
convert_images_py = dedent("""\
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
        print(f"WARNING: duplicate image basename '{base}' ({img_path} vs {seen[base]}); skipping duplicate", file=sys.stderr)
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
""")

# Python snippet that flattens the olmocr pipeline markdown output into the MDPBench
# prediction layout: result/olmocr/<image_basename>.md (matched by filename).
flatten_md_py = dedent("""\
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
        print(f"WARNING: duplicate markdown basename '{name}' ({md} vs {seen[name]}); skipping duplicate", file=sys.stderr)
        continue
    seen[name] = md
    shutil.copyfile(md, os.path.join(dst_dir, name))
    copied += 1

print(f"Copied {copied} markdown predictions to {dst_dir}")
""")

# Python snippet that patches the MDPBench end2end config to point at our downloaded
# ground truth + prediction folder. All metrics (including CDM) are kept enabled.
patch_config_py = dedent("""\
import os
import yaml

gt_path = os.path.abspath("MDPBench_dataset/MDPBench_public.json")
with open("configs/end2end.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["end2end_eval"]["dataset"]["ground_truth"]["data_path"] = gt_path
cfg["end2end_eval"]["dataset"]["prediction"]["data_path"] = "result/olmocr"

with open("configs/end2end_olmocr.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print("Wrote configs/end2end_olmocr.yaml")
print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
""")

# CDM dependency setup (Node 16 + ImageMagick 7 + full LaTeX) per MDPBench/metrics/cdm/README.md.
# Heavy and slow; baking these into a dedicated Beaker image is recommended for repeated runs.
cdm_setup_shell = dedent("""\
set -euo pipefail
echo "Installing CDM system dependencies (Node 16, ImageMagick, texlive-full)..."
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends texlive-full build-essential wget git pkg-config

# Node.js 16
if ! command -v node >/dev/null 2>&1; then
    wget -q https://registry.npmmirror.com/-/binary/node/latest-v16.x/node-v16.13.1-linux-x64.tar.gz
    tar -xf node-v16.13.1-linux-x64.tar.gz
    mkdir -p /usr/local/nodejs
    cp -r node-v16.13.1-linux-x64/* /usr/local/nodejs/
    ln -sf /usr/local/nodejs/bin/node /usr/local/bin/node
    ln -sf /usr/local/nodejs/bin/npm /usr/local/bin/npm
fi
node --version

# ImageMagick 7.1.1 from source
if ! command -v magick >/dev/null 2>&1; then
    git clone --depth 1 https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.1
    cd ImageMagick-7.1.1
    ./configure
    make -j"$(nproc)"
    make install
    ldconfig /usr/local/lib
    cd ..
fi
""")

# The full benchmark + evaluation pipeline, run inside the olmocr container.
run_mdpbench_shell = dedent("""\
bash -lc 'set -euo pipefail

# 1. Convert MDPBench images to single page PDFs
echo "Converting MDPBench images to PDFs..."
python /tmp/convert_images.py

# 2. Run the olmocr pipeline to produce markdown (pipeline manages its own vllm server)
echo "Running olmocr pipeline..."
rm -rf localworkspace
python -m olmocr.pipeline ./localworkspace --markdown --pdfs "./mdp_pdfs/*.pdf" --model "__MODEL__"

# 3. Flatten markdown into the MDPBench prediction layout
echo "Collecting markdown predictions..."
git clone --depth 1 https://github.com/Yuliang-Liu/MultimodalOCR.git
python /tmp/flatten_md.py

# 4. Set up the MDPBench eval environment (isolated venv to avoid clobbering olmocr deps)
echo "Setting up MDPBench eval environment..."
cd MultimodalOCR/MDPBench
uv venv --python 3.10 .mdpbench_venv
source .mdpbench_venv/bin/activate
uv pip install -r requirements.txt
uv pip install pyyaml

# 5. Install CDM dependencies and CDM python requirements
bash /tmp/cdm_setup.sh
if [ -f metrics/cdm/requirements.txt ]; then
    uv pip install -r metrics/cdm/requirements.txt
fi

# 6. Patch the eval config to point at our ground truth + predictions
cp /tmp/convert_images.py /tmp/flatten_md.py . 2>/dev/null || true
python /tmp/patch_config.py

# 7. Run the end-to-end evaluation and score it
echo "Running MDPBench end2end evaluation..."
python pdf_validation.py --config ./configs/end2end_olmocr.yaml

echo "Calculating scores..."
RESULT_FOLDER="result/olmocr_result"
if [ ! -d "$RESULT_FOLDER" ]; then
    # Fall back to whatever result folder pdf_validation.py produced
    RESULT_FOLDER=$(find result -maxdepth 1 -type d -name "*_result" | head -n1)
    echo "Using detected result folder: $RESULT_FOLDER"
fi
python tools/calculate_scores.py --result_folder "$RESULT_FOLDER"
'""").replace("__MODEL__", model)

# Build the command list
commands = []
if has_aws_creds:
    commands.extend([
        "mkdir -p ~/.aws",
        'echo "$AWS_CREDENTIALS_FILE" > ~/.aws/credentials'
    ])

if has_hf_token:
    commands.append('export HF_TOKEN="$HF_TOKEN"')

# Install uv for fast dependency management, then s5cmd (needed for S3 operations)
commands.append("pip install uv")
commands.append("uv pip install --system s5cmd pyyaml")

# Stage the helper python scripts onto the container
commands.append("cat > /tmp/convert_images.py << 'PYEOF'\n" + convert_images_py + "\nPYEOF")
commands.append("cat > /tmp/flatten_md.py << 'PYEOF'\n" + flatten_md_py + "\nPYEOF")
commands.append("cat > /tmp/patch_config.py << 'PYEOF'\n" + patch_config_py + "\nPYEOF")
commands.append("cat > /tmp/cdm_setup.sh << 'SHEOF'\n" + cdm_setup_shell + "\nSHEOF")

# Handle benchmark data download based on source type
if bench_path:
    if bench_path.startswith("s3://"):
        commands.append("mkdir -p ./MDPBench_dataset")
        commands.append(f"s5cmd cp {bench_path.rstrip('/')}/* ./MDPBench_dataset/")
    else:
        commands.append(f"cp -r {bench_path} ./MDPBench_dataset")
else:
    hf_download_cmd = f"hf download --repo-type dataset {bench_repo} --max-workers 4"
    if bench_branch:
        hf_download_cmd += f" --revision {bench_branch}"
    hf_download_cmd += " --local-dir ./MDPBench_dataset"
    commands.append(hf_download_cmd)

# Run the benchmark + evaluation
commands.append(run_mdpbench_shell)

# Build task spec
if '/' in image_tag:
    image_ref = image_tag
else:
    image_ref = f"{beaker_user}/{image_tag}"

task_spec_args = {
    "name": "olmocr-mdpbench",
    "image": BeakerImageSource(beaker=image_ref),
    "command": [
        "bash", "-c",
        " && ".join(commands)
    ],
    "context": BeakerTaskContext(
        priority=BeakerJobPriority["normal"],
        preemptible=True,
    ),
    "resources": BeakerTaskResources(gpu_count=1),
    "constraints": BeakerConstraints(cluster=[cluster] if cluster else ["ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2"]),
    "result": BeakerResultSpec(path="/noop-results"),
}

# Add env vars if AWS credentials or HF token exist
env_vars = []
if has_aws_creds:
    env_vars.append(BeakerEnvVar(name="AWS_CREDENTIALS_FILE", secret=aws_creds_secret))
if has_hf_token:
    env_vars.append(BeakerEnvVar(name="HF_TOKEN", secret=hf_token_secret))
if env_vars:
    task_spec_args["env_vars"] = env_vars

# Create experiment spec
experiment_spec = BeakerExperimentSpec(
    description=f"olmOCR MDPBench Run - Model: {model}, Branch: {git_branch}, Commit: {git_hash}",
    budget="ai2/oe-base",
    tasks=[BeakerTaskSpec(**task_spec_args)],
)

# Create the experiment
workload = b.experiment.create(spec=experiment_spec, workspace="ai2/olmocr")
print(f"Created MDPBench experiment: {workload.experiment.id}")
print(f"View at: https://beaker.org/ex/{workload.experiment.id}")
EOF

# Run the Python script to create the experiment
echo "Creating Beaker experiment..."

CMD="$PYTHON /tmp/run_mdpbench_experiment.py $IMAGE_TAG $BEAKER_USER $GIT_BRANCH $GIT_HASH \"$MODEL\""

if [ -n "$BENCH_BRANCH" ]; then
    echo "Using bench branch: $BENCH_BRANCH"
    CMD="$CMD --benchbranch \"$BENCH_BRANCH\""
fi

if [ -n "$BENCH_REPO" ]; then
    echo "Using bench repo: $BENCH_REPO"
    CMD="$CMD --benchrepo \"$BENCH_REPO\""
fi

if [ -n "$BENCH_PATH" ]; then
    echo "Using bench path: $BENCH_PATH"
    CMD="$CMD --benchpath \"$BENCH_PATH\""
fi

if [ -n "$CLUSTER" ]; then
    echo "Using cluster: $CLUSTER"
    CMD="$CMD --cluster $CLUSTER"
fi

eval $CMD

# Clean up temporary file
rm /tmp/run_mdpbench_experiment.py

echo "MDPBench experiment submitted successfully!"
