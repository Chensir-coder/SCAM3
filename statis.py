import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: {"train": [...], "val": [...], "test": [...]}
    if isinstance(data, dict) and any(k in data for k in ["train", "val", "test", "validation"]):
        splits = {}
        for k, v in data.items():
            key = "val" if k == "validation" else k
            if key in ["train", "val", "test"]:
                splits[key] = v
        return splits

    # Case 2: list of samples, each sample may contain a split field
    if isinstance(data, list):
        splits = defaultdict(list)
        has_split = False
        for item in data:
            split = item.get("split", None)
            if split is not None:
                has_split = True
                split = "val" if split == "validation" else split
                splits[split].append(item)

        if has_split:
            return dict(splits)

        # If no split field exists, treat all as "all"
        return {"all": data}

    raise ValueError("Unsupported JSON format.")


def flatten_splits(splits):
    all_items = []
    for split, items in splits.items():
        for item in items:
            item = dict(item)
            item["_split"] = split
            all_items.append(item)
    return all_items


def count_words(text):
    # Suitable for English referring expressions
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def get_query_level(item):
    """
    Infer unary / binary / ternary from semantic_attribute.
    Each key in semantic_attribute is treated as one geometric constraint.
    """
    attrs = item.get("semantic_attribute", {})
    if not isinstance(attrs, dict):
        return "unknown"

    n = len(attrs)
    if n <= 1:
        return "unary"
    if n == 2:
        return "binary"
    return "ternary"


def resolve_path(path_str, root):
    if path_str is None:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return Path(root) / p


def load_binary_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    return arr > 0


def skeleton_length(mask):
    """
    Compute skeleton length in pixels.
    Requires scikit-image. If unavailable, returns None.
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return None

    skel = skeletonize(mask)
    return float(skel.sum())


def print_progress(current, total, prefix="Progress", width=30):
    if total <= 0:
        return

    ratio = current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r{prefix}: [{bar}] {current}/{total} ({ratio * 100:5.1f}%)")
    sys.stderr.flush()
    if current >= total:
        sys.stderr.write("\n")


def compute_mask_stats(items, root, show_progress=True):
    area_ratios = []
    skel_lengths = []

    missing_masks = 0
    skel_available = True
    seen_mask_paths = set()
    mask_paths = []

    for item in items:
        mask_path = resolve_path(item.get("mask_path"), root)
        if mask_path in seen_mask_paths:
            continue
        seen_mask_paths.add(mask_path)
        mask_paths.append(mask_path)

    total_masks = len(mask_paths)
    for idx, mask_path in enumerate(mask_paths, start=1):
        if mask_path is None or not mask_path.exists():
            missing_masks += 1
            if show_progress:
                print_progress(idx, total_masks, prefix="Mask statistics")
            continue

        try:
            mask = load_binary_mask(mask_path)
        except Exception:
            missing_masks += 1
            if show_progress:
                print_progress(idx, total_masks, prefix="Mask statistics")
            continue

        h, w = mask.shape[:2]
        area_ratio = mask.sum() / float(h * w) * 100.0
        area_ratios.append(area_ratio)

        length = skeleton_length(mask)
        if length is None:
            skel_available = False
        else:
            skel_lengths.append(length)

        if show_progress:
            print_progress(idx, total_masks, prefix="Mask statistics")

    return {
        "avg_area_ratio": float(np.mean(area_ratios)) if area_ratios else None,
        "avg_skeleton_length": float(np.mean(skel_lengths)) if skel_lengths else None,
        "missing_masks": missing_masks,
        "skeleton_available": skel_available,
    }


def format_number(x, digits=1):
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        default="/root/code/sam3-main/anno/OSRS-Crack_all_prompt.json",
        help="Path to the unsplit dataset JSON file.",
    )
    parser.add_argument(
        "--root",
        default="/root/code/sam3-main/datasets/opencrack",
        help="Dataset root directory. Used to resolve image_path and mask_path.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print a LaTeX table for the paper.",
    )
    parser.add_argument(
        "--manual_samples",
        type=int,
        default=2000,
        help="Number of manually verified samples.",
    )
    parser.add_argument(
        "--manual_acc",
        type=float,
        default=91.0,
        help="Manual verification accuracy in percent.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress output while computing mask statistics.",
    )
    args = parser.parse_args()

    splits = load_json(args.json)
    items = flatten_splits(splits)

    # Basic counts
    unique_images = set(item.get("image_id") for item in items if item.get("image_id") is not None)
    num_images = len(unique_images)
    num_records = len(items)
    num_pairs = sum(len(item.get("prompts", [])) for item in items)

    # Query composition
    # Count by prompts, since each prompt forms an instruction-mask pair.
    query_counter = Counter()
    for item in items:
        level = get_query_level(item)
        query_counter[level] += len(item.get("prompts", []))

    unary = query_counter.get("unary", 0)
    binary = query_counter.get("binary", 0)
    ternary = query_counter.get("ternary", 0)
    known_query_total = unary + binary + ternary

    unary_pct = unary / known_query_total * 100 if known_query_total else 0
    binary_pct = binary / known_query_total * 100 if known_query_total else 0
    ternary_pct = ternary / known_query_total * 100 if known_query_total else 0

    # Expression length
    all_prompts = []
    for item in items:
        all_prompts.extend(item.get("prompts", []))

    expression_lengths = [count_words(p) for p in all_prompts]
    avg_expr_len = float(np.mean(expression_lengths)) if expression_lengths else None

    # Mask statistics
    mask_stats = compute_mask_stats(items, args.root, show_progress=not args.no_progress)

    # Console summary
    print("===== OSRS-Crack Dataset Statistics =====")
    print(f"Images: {num_images}")
    print(f"Annotation records: {num_records}")
    print(f"Instruction-mask pairs: {num_pairs}")
    print()

    print("Query composition, counted by instruction-mask pairs:")
    print(f"  Unary:   {unary} ({unary_pct:.1f}%)")
    print(f"  Binary:  {binary} ({binary_pct:.1f}%)")
    print(f"  Ternary: {ternary} ({ternary_pct:.1f}%)")
    print()

    print(f"Avg. expression length: {format_number(avg_expr_len, 1)} words")
    print(f"Avg. target area ratio: {format_number(mask_stats['avg_area_ratio'], 2)}%")
    print(f"Avg. skeleton length: {format_number(mask_stats['avg_skeleton_length'], 1)} px")
    print(f"Missing / unreadable masks: {mask_stats['missing_masks']}")

    if not mask_stats["skeleton_available"]:
        print(
            "Note: scikit-image is not installed, so skeleton length was not computed. "
            "Install it with: pip install scikit-image"
        )

    if args.latex:
        print("\n===== LaTeX Table =====")

        latex = rf"""
\begin{{table}}[t]
\centering
\caption{{Statistics and quality validation of OSRS-Crack.}}
\label{{tab:dataset_statistics}}
\begin{{tabular}}{{l r}}
\hline
Statistic & Value \\
\hline
Images & {num_images:,} \\
Instruction-mask pairs & {num_pairs:,} \\
Train / Val / Test split & 8:1:1 \\
Unary / Binary / Ternary queries & {unary_pct:.1f} / {binary_pct:.1f} / {ternary_pct:.1f}\% \\
Avg. expression length & {format_number(avg_expr_len, 1)} words \\
Avg. target area ratio & {format_number(mask_stats['avg_area_ratio'], 2)}\% \\
Avg. skeleton length & {format_number(mask_stats['avg_skeleton_length'], 1)} px \\
Manual verification & {args.manual_samples:,} samples, {args.manual_acc:.1f}\% acc. \\
\hline
\end{{tabular}}
\end{{table}}
"""
        print(latex)


if __name__ == "__main__":
    main()