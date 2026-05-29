from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.morphology import skeletonize



# ----------------------------
# Data containers
# ----------------------------

@dataclass(frozen=True)
class SampleItem:
    """One referring query / one inference result."""
    id: str
    prompt: str
    gt_mask: Path
    pred_mask: Path

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SampleItem":
        return SampleItem(
            id=d["id"],
            prompt=d.get("prompt", ""),
            gt_mask=Path(d["gt_mask"]),
            pred_mask=Path(d["pred_mask"]),
        )


@dataclass
class MetricResult:
    """Standard output container for a metric."""
    name: str
    value: float
    # Optional: extra info for debugging/plots
    details: Optional[Dict[str, Any]] = None


# ----------------------------
# Core evaluator
# ----------------------------

class SegMetricEvaluator:
    """
    Metric evaluator for referring instance segmentation on crack dataset.

    Input: json_path -> list[ {id, prompt, gt_mask, pred_mask}, ... ]
    Output: dict of scalar metrics + optional per-sample breakdown.
    """

    def __init__(
    self,
        mask_threshold: int = 0,
        max_size: Optional[int] = None,
        empty_union_iou: float = 1.0,
        cache_intermediates: bool = True,
    ):
        self.mask_threshold = mask_threshold
        self.max_size = max_size
        self.empty_union_iou = empty_union_iou
        self.cache_intermediates = cache_intermediates
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ---------
    # Public API
    # ---------

    def evaluate_all(self, json_path: str | Path) -> Dict[str, float]:
        """
        Convenience function: compute the full metric suite and return a flat dict.
        This is what you’d use to print the 'overall performance' table row.
        """
        results: Dict[str, float] = {}
        for r in self.compute_primary_metrics(json_path):
            results[r.name] = r.value
        for r in self.compute_aux_metrics(json_path):
            results[r.name] = r.value
        for r in self.compute_diagnostic_metrics(json_path):
            results[r.name] = r.value
        return results

    def compute_primary_metrics(self, json_path: str | Path) -> List[MetricResult]:
        """Main-table metrics: RefAcc@0.5, mIoU, Dice, CLDice."""
        return [
            MetricResult("RefAcc@0.5", self.refacc(json_path, thr=0.5)),
            MetricResult("mIoU", self.miou(json_path)),
            MetricResult("Dice", self.dice(json_path)),
            MetricResult("CLDice", self.cldice(json_path)),
        ]

    def compute_aux_metrics(self, json_path: str | Path) -> List[MetricResult]:
        """
        Auxiliary metrics: RefAcc@0.3/0.7 and Success–IoU AUC (optional for appendix).
        """
        return [
            MetricResult("RefAcc@0.3", self.refacc(json_path, thr=0.3)),
            MetricResult("RefAcc@0.4", self.refacc(json_path, thr=0.4)),
            MetricResult("RefAcc@0.6", self.refacc(json_path, thr=0.6)),
            MetricResult("RefAcc@0.7", self.refacc(json_path, thr=0.7)),
            MetricResult("RefAcc@0.8", self.refacc(json_path, thr=0.8)),
            MetricResult("RefAcc@0.9", self.refacc(json_path, thr=0.9)),
            MetricResult("SuccessIoU_AUC", self.success_iou_auc(json_path, num_thresholds=101)),
        ]

    def compute_diagnostic_metrics(self, json_path: str | Path) -> List[MetricResult]:
        """Diagnostic-only metrics: Boundary F-score, Skel-Recall, Skel-Precision."""
        return [
            MetricResult("BoundaryF", self.boundary_fscore(json_path, tolerance_px=2)),
            MetricResult("SkelRecall", self.skel_recall(json_path)),
            MetricResult("SkelPrecision", self.skel_precision(json_path)),
        ]

    # ----------------------------
    # Individual metric functions
    # (All accept json_path as requested)
    # ----------------------------

    def refacc(self, json_path: str | Path, thr: float) -> float:
        """RefAcc@thr = mean(IoU >= thr) over queries."""
        ious = self._get_or_compute_ious(json_path)
        return float((ious >= thr).mean())

    def miou(self, json_path: str | Path) -> float:
        """Mean IoU over queries."""
        ious = self._get_or_compute_ious(json_path)
        return float(ious.mean())

    def dice(self, json_path: str | Path) -> float:
        """Mean Dice over queries."""
        dices = self._get_or_compute_dices(json_path)
        return float(dices.mean())

    def cldice(self, json_path: str | Path) -> float:
        """Mean CLDice over queries."""
        # placeholder: compute per-sample CLDice and average
        cl = self._get_or_compute_cldices(json_path)
        return float(cl.mean())

    def boundary_fscore(self, json_path: str | Path, tolerance_px: int = 2) -> float:
        """Mean Boundary F-score over queries (with pixel tolerance)."""
        bf = self._get_or_compute_boundary_fs(json_path, tolerance_px=tolerance_px)
        return float(bf.mean())

    def skel_recall(self, json_path: str | Path) -> float:
        """Mean skeleton recall over queries."""
        sr = self._get_or_compute_skel_recalls(json_path)
        return float(sr.mean())

    def skel_precision(self, json_path: str | Path) -> float:
        """Mean skeleton precision over queries."""
        sp = self._get_or_compute_skel_precisions(json_path)
        return float(sp.mean())

    def success_iou_auc(self, json_path: str | Path, num_thresholds: int = 101) -> float:
        """
        Success–IoU AUC
        """
        ious = self._get_or_compute_ious(json_path)
        ts = np.linspace(0.0, 1.0, num_thresholds)
        S = np.array([(ious >= t).mean() for t in ts], dtype=np.float32)
        return float(np.trapz(S, ts))

    # ----------------------------
    # Data loading & intermediate computation
    # ----------------------------

    def _load_samples(self, json_path: str | Path) -> List[SampleItem]:
        """Load JSON and parse to SampleItem list."""
        json_path = Path(json_path)
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON must be a list of dicts, got: {type(data)}")
        return [SampleItem.from_dict(d) for d in data]

    def _read_mask_bool(self, mask_path: Path) -> np.ndarray:
        """
        Read a mask image and convert it to a boolean array.

        - Convert to grayscale (L)
        - Optional: isotropically downscale so that max(H, W) <= max_side (NEAREST)
        - Binarize: pixel > mask_threshold
        """
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        img = Image.open(mask_path).convert("L")

        mask = np.asarray(img, dtype=np.uint8)
        mask_bool = mask > self.mask_threshold

        if mask_bool.ndim != 2:
            raise ValueError(
                f"Mask must be 2D after conversion, got shape {mask_bool.shape} from {mask_path}"
            )
        return mask_bool


    def _read_pair(self, gt_path: Path, pred_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        gt = self._read_mask_bool(gt_path)
        pr = self._read_mask_bool(pred_path)

        # (A) Align: pred -> gt
        if pr.shape != gt.shape:
            pr_img = Image.fromarray((pr.astype(np.uint8) * 255), mode="L")
            th, tw = gt.shape  # (H, W)
            pr_img = pr_img.resize((tw, th), resample=Image.NEAREST)
            pr = (np.asarray(pr_img, dtype=np.uint8) > 0)

        # (B) Optional speed: downscale both together using ONE scale
        max_side = getattr(self, "max_size", None)
        if max_side is not None and max_side > 0:
            h, w = gt.shape
            m = max(h, w)
            if m > max_side:
                scale = max_side / float(m)
                new_h = max(1, int(round(h * scale)))
                new_w = max(1, int(round(w * scale)))

                # downscale gt/pr together with SAME target size
                gt_img = Image.fromarray((gt.astype(np.uint8) * 255), mode="L").resize(
                    (new_w, new_h), resample=Image.NEAREST
                )
                pr_img = Image.fromarray((pr.astype(np.uint8) * 255), mode="L").resize(
                    (new_w, new_h), resample=Image.NEAREST
                )
                gt = (np.asarray(gt_img, dtype=np.uint8) > 0)
                pr = (np.asarray(pr_img, dtype=np.uint8) > 0)

        return gt, pr


    def _compute_iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Single-sample IoU on boolean masks."""
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            return float(self.empty_union_iou)
        return float(inter / union)

    def _compute_dice(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Single-sample Dice on boolean masks."""
        inter = np.logical_and(pred, gt).sum()
        size_p = pred.sum()
        size_g = gt.sum()

        denom = size_p + size_g
        if denom == 0:
            # both pred and gt are empty �? perfect overlap
            return 1.0

        return float(2.0 * inter / denom)


    def _compute_cldice(self, pred: np.ndarray, gt: np.ndarray) -> float:
        def cl_score(v: np.ndarray, s: np.ndarray) -> float:
            s_sum = int(np.sum(s))
            if s_sum == 0:
                return 1.0 if int(np.sum(v)) == 0 else 0.0
            inter = int(np.sum(np.logical_and(v, s)))
            return float(inter / s_sum)

        if pred.ndim in (2, 3):
            tprec = cl_score(pred, skeletonize(gt))
            tsens = cl_score(gt, skeletonize(pred))
        else:
            raise ValueError(f"pred/gt must be 2D or 3D, got pred.ndim={pred.ndim}")

        denom = tprec + tsens
        if denom == 0.0:
            return 0.0
        return float(2.0 * tprec * tsens / denom)



    def _compute_boundary_fscore(self, pred: np.ndarray, gt: np.ndarray, tolerance_px: int = 2) -> float:
        """
        Single-sample Boundary F-score (DAVIS-style), using pixel tolerance.

        This is adapted from the DAVIS official evaluation code:
        - seg2bmap: convert mask to 1-pixel boundary map
        - dilate boundaries with a disk(tolerance_px)
        - match boundaries and compute F measure
        """
        if pred.ndim != 2 or gt.ndim != 2:
            raise ValueError(f"BoundaryF expects 2D masks, got pred.ndim={pred.ndim}, gt.ndim={gt.ndim}")
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred{pred.shape} vs gt{gt.shape}")

        from skimage.morphology import binary_dilation, disk

        pred = pred.astype(bool)
        gt = gt.astype(bool)

        # --- DAVIS seg2bmap (adapted) ---
        def seg2bmap(seg: np.ndarray) -> np.ndarray:
            """
            From a binary segmentation, compute a 1-pixel wide boundary map.
            Adapted from DAVIS seg2bmap (David Martin, 2003).
            """
            seg = seg.astype(bool)
            # ensure binary {0,1}
            seg = seg & True

            # neighbors (east, south, southeast)
            e = np.zeros_like(seg, dtype=bool)
            s = np.zeros_like(seg, dtype=bool)
            se = np.zeros_like(seg, dtype=bool)

            e[:, :-1] = seg[:, 1:]
            s[:-1, :] = seg[1:, :]
            se[:-1, :-1] = seg[1:, 1:]

            b = (seg ^ e) | (seg ^ s) | (seg ^ se)
            b[-1, :] = seg[-1, :] ^ e[-1, :]
            b[:, -1] = seg[:, -1] ^ s[:, -1]
            b[-1, -1] = False
            return b.astype(bool)

        # 1) boundary maps
        fg_boundary = seg2bmap(pred)
        gt_boundary = seg2bmap(gt)

        # 2) dilate boundaries with disk(tolerance_px)
        tol = int(max(0, tolerance_px))
        if tol == 0:
            fg_dil = fg_boundary
            gt_dil = gt_boundary
        else:
            fg_dil = binary_dilation(fg_boundary, disk(tol))
            gt_dil = binary_dilation(gt_boundary, disk(tol))

        # 3) match boundaries
        gt_match = gt_boundary & fg_dil
        fg_match = fg_boundary & gt_dil

        n_fg = int(np.sum(fg_boundary))
        n_gt = int(np.sum(gt_boundary))

        # 4) precision / recall (keep DAVIS-style edge-case behavior)
        if n_fg == 0 and n_gt > 0:
            precision = 1.0
            recall = 0.0
        elif n_fg > 0 and n_gt == 0:
            precision = 0.0
            recall = 1.0
        elif n_fg == 0 and n_gt == 0:
            precision = 1.0
            recall = 1.0
        else:
            precision = float(np.sum(fg_match)) / float(n_fg)
            recall = float(np.sum(gt_match)) / float(n_gt)

        # 5) F measure
        if precision + recall == 0.0:
            return 0.0
        return float(2.0 * precision * recall / (precision + recall))


    def _compute_skel_recall(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Single-sample Skeleton Recall on boolean masks.
        recall = |S_g �? pred| / |S_g|
        """
        if pred.ndim != 2 or gt.ndim != 2:
            raise ValueError(f"SkelRecall expects 2D masks, got pred.ndim={pred.ndim}, gt.ndim={gt.ndim}")
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred{pred.shape} vs gt{gt.shape}")

        pred = pred.astype(bool)
        gt = gt.astype(bool)

        S_g = skeletonize(gt).astype(bool)
        denom = int(S_g.sum())

        if denom == 0:
            # GT has no skeleton structure (usually gt is empty)
            # if pred is also empty, treat as perfect; else mismatch
            return 1.0 if int(pred.sum()) == 0 else 0.0

        inter = int(np.logical_and(S_g, pred).sum())
        return float(inter / denom)


    def _compute_skel_precision(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Single-sample Skeleton Precision on boolean masks.
        precision = |S_p �? gt| / |S_p|
        """
        if pred.ndim != 2 or gt.ndim != 2:
            raise ValueError(f"SkelPrecision expects 2D masks, got pred.ndim={pred.ndim}, gt.ndim={gt.ndim}")
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred{pred.shape} vs gt{gt.shape}")

        pred = pred.astype(bool)
        gt = gt.astype(bool)

        S_p = skeletonize(pred).astype(bool)
        denom = int(S_p.sum())

        if denom == 0:
            # Pred has no skeleton structure (pred is empty or tiny)
            # if gt is also empty, treat as perfect; else mismatch
            return 1.0 if int(gt.sum()) == 0 else 0.0

        inter = int(np.logical_and(S_p, gt).sum())
        return float(inter / denom)



    # ---- caching helpers ----

    def _cache_key(self, json_path: str | Path) -> str:
        return str(Path(json_path).resolve())

    def _get_or_compute_ious(self, json_path: str | Path) -> np.ndarray:
        key = self._cache_key(json_path)
        if self.cache_intermediates and key in self._cache and "ious" in self._cache[key]:
            return self._cache[key]["ious"]

        samples = self._load_samples(json_path)
        ious: List[float] = []
        # Optional store per-sample info for debug
        for s in tqdm(samples, desc="compute_ious"):
            gt, pr = self._read_pair(s.gt_mask, s.pred_mask)  # �� max_side=None

            ious.append(self._compute_iou(pr, gt))

        arr = np.array(ious, dtype=np.float32)

        if self.cache_intermediates:
            self._cache.setdefault(key, {})["ious"] = arr
        return arr

    def _get_or_compute_dices(self, json_path: str | Path) -> np.ndarray:
        key = self._cache_key(json_path)
        if self.cache_intermediates and key in self._cache and "dices" in self._cache[key]:
            return self._cache[key]["dices"]

        samples = self._load_samples(json_path)
        dices: List[float] = []
        for s in tqdm(samples, desc="compute_dices"):
            gt, pr = self._read_pair(s.gt_mask, s.pred_mask)
            dices.append(self._compute_dice(pr, gt))

        arr = np.array(dices, dtype=np.float32)
        if self.cache_intermediates:
            self._cache.setdefault(key, {})["dices"] = arr
        return arr

    def _get_or_compute_cldices(self, json_path: str | Path) -> np.ndarray:
        """Compute per-sample CLDice."""
        key = self._cache_key(json_path)
        if self.cache_intermediates and key in self._cache and "cldices" in self._cache[key]:
            return self._cache[key]["cldices"]

        samples = self._load_samples(json_path)
        cldices: List[float] = []
        for s in tqdm(samples, desc="compute_cldices"):
            gt, pr = self._read_pair(s.gt_mask, s.pred_mask)
            cldices.append(self._compute_cldice(pr, gt))  # TODO: implement

        arr = np.array(cldices, dtype=np.float32)
        if self.cache_intermediates:
            self._cache.setdefault(key, {})["cldices"] = arr
        return arr


    def _get_or_compute_boundary_fs(self, json_path: str | Path, tolerance_px: int) -> np.ndarray:
        """Compute per-sample Boundary F-score."""
        key = self._cache_key(json_path)

        # Important: cache key should include tolerance
        cache_name = f"boundary_fs@tol={tolerance_px}"
        if self.cache_intermediates and key in self._cache and cache_name in self._cache[key]:
            return self._cache[key][cache_name]

        samples = self._load_samples(json_path)
        bfs: List[float] = [] 
        for s in tqdm(samples, desc=f"compute_boundary_fs(tol={tolerance_px})"):
            gt, pr = self._read_pair(s.gt_mask, s.pred_mask)
            bfs.append(self._compute_boundary_fscore(pr, gt, tolerance_px=tolerance_px))  # TODO

        arr = np.array(bfs, dtype=np.float32)
        if self.cache_intermediates:
            self._cache.setdefault(key, {})[cache_name] = arr
        return arr


    def _get_or_compute_skel_recalls(self, json_path: str | Path) -> np.ndarray:
        """Compute per-sample skeleton recall."""
        key = self._cache_key(json_path)
        if self.cache_intermediates and key in self._cache and "skel_recalls" in self._cache[key]:
            return self._cache[key]["skel_recalls"]

        samples = self._load_samples(json_path)
        recalls: List[float] = []
        for s in tqdm(samples, desc="compute_skel_recalls"):
            gt, pr = self._read_pair(s.gt_mask, s.pred_mask)
            recalls.append(self._compute_skel_recall(pr, gt))  # TODO

        arr = np.array(recalls, dtype=np.float32)
        if self.cache_intermediates:
            self._cache.setdefault(key, {})["skel_recalls"] = arr
        return arr


    def _get_or_compute_skel_precisions(self, json_path: str | Path) -> np.ndarray:
        """Compute per-sample skeleton precision."""
        key = self._cache_key(json_path)
        if self.cache_intermediates and key in self._cache and "skel_precisions" in self._cache[key]:
            return self._cache[key]["skel_precisions"]

        samples = self._load_samples(json_path)
        precisions: List[float] = []
        for s in tqdm(samples, desc="compute_skel_precisions"):
            gt, pr = self._read_pair(s.gt_mask, s.pred_mask)
            precisions.append(self._compute_skel_precision(pr, gt))  # TODO

        arr = np.array(precisions, dtype=np.float32)
        if self.cache_intermediates:
            self._cache.setdefault(key, {})["skel_precisions"] = arr
        return arr

    def precompute_all_metrics(
        self,
        json_path: str | Path,
        *,
        tolerance_px: int = 2,
        num_thresholds: int = 101,
        compute_auc: bool = True,
    ) -> Dict[str, float]:
        """
        One-pass IO: read each (gt, pred) mask once, compute all per-sample metrics once,
        cache arrays into self._cache, and return a flat dict of overall metrics.

        IMPORTANT:
        - Does NOT modify existing metric methods.
        - After calling this, refacc/miou/dice/cldice/boundary_fscore/skel_* will
        reuse cached arrays and avoid repeated IO.

        Returns:
            A dict of overall scalar metrics (same style as evaluate_all output).
        """
        key = self._cache_key(json_path)

        # If already precomputed for this tolerance, just return overall metrics quickly
        cache_name_bf = f"boundary_fs@tol={tolerance_px}"
        if (
            self.cache_intermediates
            and key in self._cache
            and "ious" in self._cache[key]
            and "dices" in self._cache[key]
            and "cldices" in self._cache[key]
            and cache_name_bf in self._cache[key]
            and "skel_recalls" in self._cache[key]
            and "skel_precisions" in self._cache[key]
        ):
            # Use existing public API to assemble scalars (no extra IO)
            out = self.evaluate_all(json_path)
            return out

        samples = self._load_samples(json_path)

        ious: List[float] = []
        dices: List[float] = []
        cldices: List[float] = []
        bfs: List[float] = []
        srec: List[float] = []
        sprec: List[float] = []

        for s in tqdm(samples, desc=f"precompute_all_metrics(tol={tolerance_px})"):
            gt, pr = self._read_pair(s.gt_mask, s.pred_mask)  # �� max_side=None

            # compute everything once
            ious.append(self._compute_iou(pr, gt))
            dices.append(self._compute_dice(pr, gt))
            cldices.append(self._compute_cldice(pr, gt))
            bfs.append(self._compute_boundary_fscore(pr, gt, tolerance_px=tolerance_px))
            srec.append(self._compute_skel_recall(pr, gt))
            sprec.append(self._compute_skel_precision(pr, gt))

        # convert to arrays
        arr_ious = np.asarray(ious, dtype=np.float32)
        arr_dices = np.asarray(dices, dtype=np.float32)
        arr_cldices = np.asarray(cldices, dtype=np.float32)
        arr_bfs = np.asarray(bfs, dtype=np.float32)
        arr_srec = np.asarray(srec, dtype=np.float32)
        arr_sprec = np.asarray(sprec, dtype=np.float32)

        # cache
        if self.cache_intermediates:
            self._cache.setdefault(key, {})["ious"] = arr_ious
            self._cache[key]["dices"] = arr_dices
            self._cache[key]["cldices"] = arr_cldices
            self._cache[key][cache_name_bf] = arr_bfs
            self._cache[key]["skel_recalls"] = arr_srec
            self._cache[key]["skel_precisions"] = arr_sprec

        # produce final scalars (no extra IO)
        results = self.evaluate_all(json_path)

        # Optionally compute AUC here and add into dict (still no extra IO)
        if compute_auc:
            ts = np.linspace(0.0, 1.0, num_thresholds)
            S = np.array([(arr_ious >= t).mean() for t in ts], dtype=np.float32)
            results["SuccessIoU_AUC"] = float(np.trapz(S, ts))

        return results
    
    def save_results(
        self,
        results: Dict[str, float],
        out_path: str | Path,
        *,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save overall metric results to a JSON file.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "metrics": results,
        }
        if extra_meta is not None:
            payload["meta"] = extra_meta

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return out_path

    def precompute_all_metrics_and_save(
        self,
        json_path: str | Path,
        out_path: str | Path,
        *,
        tolerance_px: int = 2,
        num_thresholds: int = 101,
        compute_auc: bool = True,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Precompute all metrics in one pass, then save overall results to JSON.
        """
        results = self.precompute_all_metrics(
            json_path,
            tolerance_px=tolerance_px,
            num_thresholds=num_thresholds,
            compute_auc=compute_auc,
        )

        meta = dict(extra_meta or {})
        meta.update(
            {
                "json_path": str(Path(json_path)),
                "tolerance_px": tolerance_px,
                "num_thresholds": num_thresholds,
                "compute_auc": compute_auc,
            }
        )

        self.save_results(results, out_path, extra_meta=meta)
        return results


