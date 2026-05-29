#!/usr/bin/env python3

from pathlib import Path

from pycocotools import mask as mask_util

from sam3.train.data.osrs_crack_loader import OSRS_CRACK_FROM_JSON


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    ann_file = (
        repo_root
        / "datasets"
        / "opencrack_processed"
        / "OSRS-Crack_train_expanded.json"
    )
    cache_file = (
        repo_root
        / "datasets"
        / "opencrack_processed"
        / "OSRS-Crack_train_mask_cache.json"
    )

    if not ann_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    if not cache_file.is_file():
        raise FileNotFoundError(f"Mask cache file not found: {cache_file}")

    loader = OSRS_CRACK_FROM_JSON(str(ann_file), str(cache_file))

    queries, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(0)
    rle = annotations[0]["segmentation"]
    mask = mask_util.decode(rle)

    print("query:", queries[0]["query_text"])
    print("mask shape:", mask.shape)
    print("mask dtype:", mask.dtype)
    print("mask sum:", int(mask.sum()))
    print("bbox:", annotations[0]["bbox"])
    print("ok:", mask.ndim == 2 and mask.sum() > 0)


if __name__ == "__main__":
    main()
