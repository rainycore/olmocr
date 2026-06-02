"""Flatten the olmOCR pipeline markdown output into the MDPBench prediction layout.

Run from the working dir (cwd) that contains localworkspace/ and MultimodalOCR/.
Copies localworkspace/markdown/**/<name>.md -> MultimodalOCR/MDPBench/result/olmocr/<name>.md
(MDPBench matches predictions to ground truth by filename).
"""
import glob
import os
import shutil
import sys

SRC_GLOB = "localworkspace/markdown/**/*.md"
DST_DIR = "MultimodalOCR/MDPBench/result/olmocr"


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    md_files = sorted(glob.glob(SRC_GLOB, recursive=True))
    print(f"Found {len(md_files)} markdown outputs from the pipeline")

    seen = {}
    copied = 0
    for md in md_files:
        name = os.path.basename(md)
        if name in seen:
            print(f"WARNING: duplicate markdown basename '{name}' ({md} vs {seen[name]}); skipping", file=sys.stderr)
            continue
        seen[name] = md
        shutil.copyfile(md, os.path.join(DST_DIR, name))
        copied += 1

    print(f"Copied {copied} markdown predictions to {DST_DIR}")
    if copied == 0:
        sys.exit("ERROR: no markdown predictions were produced by the pipeline")


if __name__ == "__main__":
    main()
