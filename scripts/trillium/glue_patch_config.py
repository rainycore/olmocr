"""Patch MDPBench's end2end.yaml to point at our ground truth + predictions.

Run from inside MultimodalOCR/MDPBench. Reads configs/end2end.yaml, sets the ground-truth
path (from the MDP_GT_PATH env var) and the prediction folder (result/olmocr), and -- unless
MDP_KEEP_CDM=1 -- drops the CDM formula metric (heavy deps not available on this cluster).
Writes configs/end2end_olmocr.yaml.
"""
import os
import sys

import yaml

GT_PATH = os.environ.get("MDP_GT_PATH")
KEEP_CDM = os.environ.get("MDP_KEEP_CDM") == "1"

if not GT_PATH or not os.path.isfile(GT_PATH):
    sys.exit(f"ERROR: MDP_GT_PATH not set or missing: {GT_PATH!r}")

with open("configs/end2end.yaml") as f:
    cfg = yaml.safe_load(f)

ds = cfg["end2end_eval"]["dataset"]
ds["ground_truth"]["data_path"] = GT_PATH
ds["prediction"]["data_path"] = "result/olmocr"

if not KEEP_CDM:
    metrics = cfg["end2end_eval"]["metrics"].get("display_formula", {})
    if "metric" in metrics:
        metrics["metric"] = [m for m in metrics["metric"] if m != "CDM"]

with open("configs/end2end_olmocr.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("Wrote configs/end2end_olmocr.yaml")
print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
