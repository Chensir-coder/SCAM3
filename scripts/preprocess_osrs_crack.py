#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess OSRS-Crack annotations for SAM3 fine-tuning.

This script generates:
  1. OSRS-Crack_{split}_expanded.json
     - One item = one image-query-mask sample.
     - Every prompt in "prompts" becomes an independent training sample.

  2. OSRS-Crack_{split}_mask_cache.json
     - One item = one unique mask_path.
     - Stores bbox, area, mask area statistics, and COCO-style compressed RLE.

Expected original data layout:

datasets/osrs_crack/
├── OSRS-Crack_train.json
├── OSRS-Crack_val.json
├── OSRS-Crack_test.json
├── images/
├── masks/
└── highlighted/
"""

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from pycocotools import mask as mask_util
except ImportError as exc:
    raise ImportError(
        "pycocotools is required. Please install it with:\n"
        "  pip install pycocotools"
    ) from exc


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def sanitize_id(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def read_binary_mask(mask_path: Path) -> np.ndarray:
    """
    Read mask PNG as binary mask.

    Returns:
        mask: bool array, shape [H, W]
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    return mask > 0


def binary_mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Convert binary mask to COCO-style compressed RLE.

    Args:
        mask: bool or uint8 array, shape [H, W]

    Returns:
        rle: {"size": [H, W], "counts": "..."}
    """
    mask_uint8 = mask.astype(np.uint8)
    mask_fortran = np.asfortranarray(mask_uint8)
    rle = mask_util.encode(mask_fortran)

    # pycocotools returns bytes; JSON needs str.
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")

    return rle


def rle_to_binary_mask(rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode COCO-style compressed RLE back to binary mask.
    Used only for sanity checking.
    """
    rle_copy = copy.deepcopy(rle)

    if isinstance(rle_copy["counts"], str):
        rle_copy["counts"] = rle_copy["counts"].encode("utf-8")

    decoded = mask_util.decode(rle_copy)
    return decoded.astype(bool)


def mask_to_bbox_xywh_norm(mask: np.ndarray) -> Tuple[List[float], float, int, float]:
    """
    Compute normalized xywh bbox from a binary mask.

    Returns:
        bbox_norm:
            [x, y, w, h], normalized by image width/height.

        bbox_area_norm:
            normalized bbox area, i.e. bbox_w_norm * bbox_h_norm.
            This mirrors the behavior in SAM3's COCO_FROM_JSON, where
            annotation["area"] is set to bbox[2] * bbox[3].

        mask_area_pixels:
            number of foreground pixels.

        mask_area_ratio:
            foreground pixels / total pixels.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    h, w = mask.shape
    ys, xs = np.where(mask)

    mask_area_pixels = int(mask.sum())
    mask_area_ratio = float(mask_area_pixels) / float(h * w)

    if len(xs) == 0:
        raise ValueError("Empty mask: no foreground pixels found.")

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    # xywh with inclusive pixel support.
    # If the object is one pixel wide, width should be 1, not 0.
    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1

    bbox_norm = [
        float(x_min) / float(w),
        float(y_min) / float(h),
        float(box_w) / float(w),
        float(box_h) / float(h),
    ]

    bbox_area_norm = bbox_norm[2] * bbox_norm[3]

    return bbox_norm, bbox_area_norm, mask_area_pixels, mask_area_ratio


def check_image_mask_size(image_path: Path, mask: np.ndarray) -> Tuple[int, int]:
    """
    Check that image and mask have the same width/height.

    Returns:
        width, height
    """
    image = Image.open(image_path)
    image_w, image_h = image.size

    mask_h, mask_w = mask.shape

    if image_w != mask_w or image_h != mask_h:
        raise ValueError(
            "Image/mask size mismatch:\n"
            f"  image: {image_path} size=({image_w}, {image_h})\n"
            f"  mask : size=({mask_w}, {mask_h})"
        )

    return image_w, image_h


def build_expanded_samples(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expand each original annotation with multiple prompts into multiple samples.

    One output item = one image-query-mask sample.
    """
    expanded = []

    for item_idx, item in enumerate(raw_data):
        prompts = item.get("prompts", None)

        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError(
                f"Item index {item_idx} has no valid 'prompts' list: {item}"
            )

        source_item_id = item.get("id", f"item_{item_idx}")
        source_item_id = sanitize_id(source_item_id)

        for prompt_idx, prompt in enumerate(prompts):
            if not isinstance(prompt, str) or prompt.strip() == "":
                raise ValueError(
                    f"Empty prompt at item index {item_idx}, prompt index {prompt_idx}"
                )

            sample = {}

            # Preserve useful metadata.
            for key, value in item.items():
                if key == "prompts":
                    continue
                sample[key] = value

            sample["sample_id"] = f"{source_item_id}__p{prompt_idx:02d}"
            sample["source_item_id"] = item.get("id", str(item_idx))
            sample["source_index"] = item_idx
            sample["prompt_index"] = prompt_idx
            sample["query_text"] = prompt.strip()

            expanded.append(sample)

    return expanded


def build_mask_cache(
    raw_data: List[Dict[str, Any]],
    root: Path,
    verify_rle: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Build cache for each unique mask_path.

    The cache stores:
      - image_path
      - width / height
      - normalized xywh bbox
      - area used by SAM3
      - mask foreground statistics
      - COCO-style compressed RLE
    """
    cache: Dict[str, Dict[str, Any]] = {}
    mask_to_image: Dict[str, str] = {}

    for item_idx, item in enumerate(raw_data):
        image_rel = item.get("image_path", None)
        mask_rel = item.get("mask_path", None)

        if image_rel is None:
            raise ValueError(f"Item index {item_idx} missing 'image_path'")
        if mask_rel is None:
            raise ValueError(f"Item index {item_idx} missing 'mask_path'")

        image_path = root / image_rel
        mask_path = root / mask_rel

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if mask_rel in cache:
            # Same mask appears in multiple query annotations.
            # Make sure it always points to the same image.
            if mask_to_image[mask_rel] != image_rel:
                raise ValueError(
                    f"Same mask_path appears with different image_path:\n"
                    f"  mask_path: {mask_rel}\n"
                    f"  previous image_path: {mask_to_image[mask_rel]}\n"
                    f"  current  image_path: {image_rel}"
                )
            continue

        mask = read_binary_mask(mask_path)
        width, height = check_image_mask_size(image_path, mask)

        bbox_norm, bbox_area_norm, mask_area_pixels, mask_area_ratio = (
            mask_to_bbox_xywh_norm(mask)
        )

        rle = binary_mask_to_rle(mask)

        if verify_rle:
            decoded = rle_to_binary_mask(rle)
            if not np.array_equal(mask, decoded):
                raise ValueError(f"RLE verification failed for mask: {mask_path}")

        cache[mask_rel] = {
            "image_path": image_rel,
            "height": height,
            "width": width,

            # Normalized xywh bbox. This is what our SAM3 loader will feed as
            # annotation["bbox"].
            "bbox": bbox_norm,

            # For compatibility with SAM3's COCO_FROM_JSON behavior:
            # annotation["area"] = bbox[2] * bbox[3].
            "area": bbox_area_norm,

            # Extra statistics for debugging / analysis.
            "bbox_area_norm": bbox_area_norm,
            "mask_area_pixels": mask_area_pixels,
            "mask_area_ratio": mask_area_ratio,

            # COCO-style compressed RLE.
            "segmentation": rle,
        }

        mask_to_image[mask_rel] = image_rel

    return cache


def summarize(
    split: str,
    raw_data: List[Dict[str, Any]],
    expanded: List[Dict[str, Any]],
    mask_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    semantic_counter = Counter()
    image_paths = set()
    mask_paths = set()

    for item in raw_data:
        image_paths.add(item.get("image_path", ""))
        mask_paths.add(item.get("mask_path", ""))
        semantic_counter[item.get("semantic_type", "UNKNOWN")] += 1

    mask_area_ratios = [v["mask_area_ratio"] for v in mask_cache.values()]
    bbox_area_norms = [v["bbox_area_norm"] for v in mask_cache.values()]

    def safe_min(values):
        return float(min(values)) if values else None

    def safe_max(values):
        return float(max(values)) if values else None

    def safe_mean(values):
        return float(sum(values) / len(values)) if values else None

    stats = {
        "split": split,
        "num_raw_items": len(raw_data),
        "num_expanded_samples": len(expanded),
        "num_unique_images": len(image_paths),
        "num_unique_masks": len(mask_cache),
        "semantic_type_counts": dict(semantic_counter),
        "mask_area_ratio": {
            "min": safe_min(mask_area_ratios),
            "max": safe_max(mask_area_ratios),
            "mean": safe_mean(mask_area_ratios),
        },
        "bbox_area_norm": {
            "min": safe_min(bbox_area_norms),
            "max": safe_max(bbox_area_norms),
            "mean": safe_mean(bbox_area_norms),
        },
    }

    return stats


def process_split(
    root: Path,
    split: str,
    ann_pattern: str,
    output_dir: Path,
    verify_rle: bool,
    overwrite: bool,
) -> None:
    ann_path = root / ann_pattern.format(split=split)

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    expanded_path = output_dir / f"OSRS-Crack_{split}_expanded.json"
    mask_cache_path = output_dir / f"OSRS-Crack_{split}_mask_cache.json"
    stats_path = output_dir / f"OSRS-Crack_{split}_preprocess_stats.json"

    if not overwrite:
        for p in [expanded_path, mask_cache_path, stats_path]:
            if p.exists():
                raise FileExistsError(
                    f"Output file already exists: {p}\n"
                    "Use --overwrite if you want to overwrite it."
                )

    print(f"\n[Split: {split}]")
    print(f"Loading: {ann_path}")

    raw_data = load_json(ann_path)
    if not isinstance(raw_data, list):
        raise ValueError(f"Expected list JSON in {ann_path}, got {type(raw_data)}")

    print("Building expanded samples...")
    expanded = build_expanded_samples(raw_data)

    print("Building mask cache...")
    mask_cache = build_mask_cache(raw_data, root=root, verify_rle=verify_rle)

    stats = summarize(split, raw_data, expanded, mask_cache)

    print("Saving outputs...")
    save_json(expanded, expanded_path)
    save_json(mask_cache, mask_cache_path)
    save_json(stats, stats_path)

    print(f"Saved expanded     : {expanded_path}")
    print(f"Saved mask cache   : {mask_cache_path}")
    print(f"Saved stats        : {stats_path}")

    print("Summary:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess OSRS-Crack annotations for SAM3 fine-tuning."
    )

    parser.add_argument(
        "--root",
        type=str,
        help="Dataset root, e.g. /path/to/datasets/osrs_crack",
    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process. Default: train val test",
    )

    parser.add_argument(
        "--ann-pattern",
        type=str,
        default="OSRS-Crack_{split}.json",
        help=(
            "Annotation filename pattern under root. "
            "Default: OSRS-Crack_{split}.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Default: same as --root. "
            "Generated files will be saved here."
        ),
    )

    parser.add_argument(
        "--no-verify-rle",
        action="store_true",
        help="Disable RLE decode verification.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else root
    )

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    print(f"Dataset root: {root}")
    print(f"Output dir  : {output_dir}")

    for split in args.splits:
        process_split(
            root=root,
            split=split,
            ann_pattern=args.ann_pattern,
            output_dir=output_dir,
            verify_rle=not args.no_verify_rle,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()