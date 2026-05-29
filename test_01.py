#!/usr/bin/env python3

from pathlib import Path

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

    loader = OSRS_CRACK_FROM_JSON(
        annotation_file=str(ann_file),
        mask_cache_file=str(cache_file),
    )

    print("num datapoints:", len(loader.getDatapointIds()))

    idx = 0
    print("\nImages:")
    print(loader.loadImagesFromDatapoint(idx))

    queries, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(idx)

    print("\nQueries:")
    print(queries)

    print("\nAnnotations:")
    ann = annotations[0]
    print(
        {
            "image_id": ann["image_id"],
            "bbox": ann["bbox"],
            "area": ann["area"],
            "object_id": ann["object_id"],
            "id": ann["id"],
            "source": ann["source"],
            "segmentation_size": ann["segmentation"]["size"],
            "segmentation_counts_type": type(
                ann["segmentation"]["counts"]
            ).__name__,
        }
    )


if __name__ == "__main__":
    main()
