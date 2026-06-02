"""Convert MDPBench images into single page PDFs for the olmOCR pipeline.

Run from the working dir (cwd) that contains MDPBench_dataset/. Produces ./mdp_pdfs/<basename>.pdf,
preserving the image basename so predictions can be matched to ground truth by filename.
"""
import glob
import os
import sys

from olmocr.image_utils import convert_image_to_pdf_bytes

IMG_DIR = "MDPBench_dataset/MDPBench_img_public"
OUT_DIR = "mdp_pdfs"
EXTS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG", "*.tif", "*.tiff", "*.webp", "*.bmp")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    images = []
    for ext in EXTS:
        images.extend(glob.glob(os.path.join(IMG_DIR, "**", ext), recursive=True))
    images = sorted(set(images))
    print(f"Found {len(images)} MDPBench images under {IMG_DIR}")

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
        with open(os.path.join(OUT_DIR, base + ".pdf"), "wb") as f:
            f.write(pdf_bytes)
        written += 1

    print(f"Wrote {written} PDFs to {OUT_DIR}")
    if written == 0:
        sys.exit("ERROR: no PDFs were produced")


if __name__ == "__main__":
    main()
