# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
OSRS-Crack JSON loader for SAM3 image fine-tuning.

This loader adapts the preprocessed OSRS-Crack files to the same interface used by
COCO_FROM_JSON in sam3/train/data/coco_json_loaders.py.

Expected files:

1. Expanded annotation file:
   OSRS-Crack_train_expanded.json

   [
     {
       "sample_id": "...__p00",
       "image_id": "CrackSeg_00001",
       "image_path": "images/CrackSeg/CrackSeg_00001.png",
       "mask_path": "masks/CrackSeg/CrackSeg_00001_01.png",
       "query_text": "the thin crack",
       "semantic_type": "geometry",
       ...
     }
   ]

2. Mask cache file:
   OSRS-Crack_train_mask_cache.json

   {
     "masks/CrackSeg/CrackSeg_00001_01.png": {
       "image_path": "images/CrackSeg/CrackSeg_00001.png",
       "height": 512,
       "width": 512,
       "bbox": [x, y, w, h],   # normalized xywh
       "area": 0.0123,
       "segmentation": {
         "size": [512, 512],
         "counts": "..."
       }
     }
   }

The SAM3 dataset will call:

    loader = OSRS_CRACK_FROM_JSON(annotation_file)
    ids = loader.getDatapointIds()
    images = loader.loadImagesFromDatapoint(idx)
    queries, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(idx)

Each datapoint corresponds to one image-query-mask sample.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _derive_mask_cache_file(annotation_file: str) -> str:
    """
    Derive mask cache path from expanded annotation file.

    Example:
        OSRS-Crack_train_expanded.json
            -> OSRS-Crack_train_mask_cache.json
    """
    dirname = os.path.dirname(annotation_file)
    basename = os.path.basename(annotation_file)

    if basename.endswith("_expanded.json"):
        cache_basename = basename.replace("_expanded.json", "_mask_cache.json")
    else:
        stem, _ = os.path.splitext(basename)
        cache_basename = f"{stem}_mask_cache.json"

    return os.path.join(dirname, cache_basename)


def _safe_int(value: Any, fallback: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


class OSRS_CRACK_FROM_JSON:
    """
    Loader for OSRS-Crack referring segmentation training.

    This class mimics COCO_FROM_JSON's API, but the unit of one datapoint is:
        one image + one query_text + one target mask.

    It assumes prompts have already been expanded, so no random prompt sampling
    happens here.
    """

    def __init__(
        self,
        annotation_file: str,
        mask_cache_file: Optional[str] = None,
        original_cat_id: int = 1,
        source_name: str = "osrs_crack",
        strict: bool = True,
    ) -> None:
        """
        Args:
            annotation_file:
                Path to OSRS-Crack_{split}_expanded.json.

            mask_cache_file:
                Path to OSRS-Crack_{split}_mask_cache.json.
                If None, it is derived automatically from annotation_file.

            original_cat_id:
                Category id used for SAM3 metadata.
                Since OSRS-Crack has referring expressions rather than COCO
                categories, we use a single pseudo category id by default.

            source_name:
                Source string stored in each annotation.

            strict:
                If True, run basic validation on files and fields.
        """
        self.annotation_file = annotation_file
        self.mask_cache_file = (
            mask_cache_file
            if mask_cache_file is not None
            else _derive_mask_cache_file(annotation_file)
        )

        self.original_cat_id = int(original_cat_id)
        self.source_name = source_name
        self.strict = strict

        self._samples: List[Dict[str, Any]] = _load_json(self.annotation_file)
        self._mask_cache: Dict[str, Dict[str, Any]] = _load_json(self.mask_cache_file)

        if self.strict:
            self._validate()

        print(
            "[OSRS_CRACK_FROM_JSON] "
            f"annotation_file={self.annotation_file}, "
            f"mask_cache_file={self.mask_cache_file}, "
            f"samples={len(self._samples)}, "
            f"unique_masks={len(self._mask_cache)}"
        )

    def _validate(self) -> None:
        if not isinstance(self._samples, list):
            raise ValueError(
                f"Expected annotation file to contain a list, got {type(self._samples)}"
            )

        if not isinstance(self._mask_cache, dict):
            raise ValueError(
                f"Expected mask cache file to contain a dict, got {type(self._mask_cache)}"
            )

        required_sample_keys = ["image_path", "mask_path", "query_text"]
        required_cache_keys = ["height", "width", "bbox", "area", "segmentation"]

        for idx, sample in enumerate(self._samples):
            for key in required_sample_keys:
                if key not in sample:
                    raise ValueError(f"Sample {idx} missing key '{key}': {sample}")

            if not isinstance(sample["query_text"], str) or sample["query_text"] == "":
                raise ValueError(f"Sample {idx} has empty query_text: {sample}")

            mask_path = sample["mask_path"]
            if mask_path not in self._mask_cache:
                raise ValueError(
                    f"Sample {idx} refers to mask_path not found in mask cache:\n"
                    f"  mask_path={mask_path}"
                )

            cache = self._mask_cache[mask_path]
            for key in required_cache_keys:
                if key not in cache:
                    raise ValueError(
                        f"Mask cache for '{mask_path}' missing key '{key}'"
                    )

            if "image_path" in cache and cache["image_path"] != sample["image_path"]:
                raise ValueError(
                    "image_path mismatch between expanded sample and mask cache:\n"
                    f"  sample image_path={sample['image_path']}\n"
                    f"  cache  image_path={cache['image_path']}\n"
                    f"  mask_path={mask_path}"
                )

            bbox = cache["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(
                    f"Mask cache bbox must be a list of 4 floats: {mask_path}"
                )

            if any((v < 0.0 or v > 1.0) for v in bbox):
                raise ValueError(
                    f"Mask cache bbox values should be normalized to [0, 1]: "
                    f"{mask_path}, bbox={bbox}"
                )

            seg = cache["segmentation"]
            if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
                raise ValueError(
                    f"Mask cache segmentation must be COCO-style RLE: {mask_path}"
                )

    def getDatapointIds(self) -> List[int]:
        """
        Return all datapoint indices.

        Since the expanded json already has one item per image-query-mask pair,
        one epoch over this loader means one pass over all expanded samples.
        """
        return list(range(len(self._samples)))

    def loadImagesFromDatapoint(self, idx: int) -> List[Dict[str, Any]]:
        """
        Return image metadata for one datapoint.

        SAM3's CustomCocoDetectionAPI will join:
            img_folder + file_name
        to actually read the image.
        """
        sample = self._samples[idx]
        cache = self._mask_cache[sample["mask_path"]]

        # original_img_id can be a string in OSRS-Crack, for example "CrackSeg_00001".
        # Sam3ImageDataset will try int(...) for eval metadata and fallback to -1
        # if it is not numeric. That is OK.
        original_img_id = sample.get("image_id", idx)

        images = [
            {
                "id": 0,
                "file_name": sample["image_path"],
                "original_img_id": original_img_id,
                # coco_img_id is used only for eval metadata. Keep it numeric.
                "coco_img_id": _safe_int(original_img_id, fallback=idx),
                # Extra metadata, not required by SAM3, but useful for debugging.
                "height": cache["height"],
                "width": cache["width"],
            }
        ]

        return images

    def loadQueriesAndAnnotationsFromDatapoint(
        self, idx: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Return queries and annotations for one datapoint.

        For OSRS-Crack, each expanded sample has exactly:
            - one query
            - one target object annotation
        """
        sample = self._samples[idx]
        mask_path = sample["mask_path"]
        cache = self._mask_cache[mask_path]

        bbox = torch.tensor(cache["bbox"], dtype=torch.float32)

        # Make a shallow copy so downstream transforms can modify it safely.
        # Important: pycocotools can decode compressed RLE where counts is str.
        segmentation = dict(cache["segmentation"])

        annotation = {
            "image_id": 0,
            "bbox": bbox,  # normalized xywh
            "area": float(cache["area"]),
            "segmentation": segmentation,
            "object_id": 0,
            "is_crowd": False,
            "id": 0,
            "source": self.source_name,
        }

        query = {
            "id": 0,
            "original_cat_id": self.original_cat_id,
            "object_ids_output": [0],
            "query_text": sample["query_text"],
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
            "is_pixel_exhaustive": True,
        }

        return [query], [annotation]