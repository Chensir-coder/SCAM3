import os
from contextlib import nullcontext

import torch
from PIL import Image
import numpy as np
import matplotlib
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from overall_performance_eval.model_eval_base import OSRS_Crack_Inference_Base  # 替换成你的模块路径

class Sam3_Eval(OSRS_Crack_Inference_Base):
    def __init__(self, model_name, model_path, work_dir="output"):
        super().__init__(model_name, model_path)
        self.work_dir = os.path.join(work_dir, model_name, "image")
        os.makedirs(os.path.join(work_dir, model_name, "image"),exist_ok=True)


    def load_model(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            # SAM3 image examples enable TF32 and bfloat16 autocast for inference.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model = build_sam3_image_model(
            checkpoint_path=model_path,
            device=self.device,
            load_from_HF=False,
        )
        processor = Sam3Processor(model, device=self.device)
        return {"processor": processor, "model": model}

    @staticmethod
    def _masks_to_numpy(masks):
        if masks is None:
            return np.zeros((0, 0, 0), dtype=bool)

        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu()
            if masks.dtype != torch.bool:
                masks = masks > 0.5
            masks = masks.numpy()
        else:
            masks = np.asarray(masks)

        masks = np.squeeze(masks)
        if masks.ndim == 2:
            masks = masks[None, :, :]
        elif masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        elif masks.ndim != 3:
            raise ValueError(f"Unexpected masks shape: {masks.shape}")

        return masks.astype(bool)

    def _autocast_context(self):
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def overlay_masks(image, masks, alpha=0.5):
        """叠加掩码到原图"""
        image = image.convert("RGBA")
        masks = Sam3_Eval._masks_to_numpy(masks)
        n_masks = masks.shape[0]
        if n_masks == 0:
            return image

        alpha = alpha if alpha <= 1 else alpha / 255
        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
        colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

        for mask, color in zip(masks, colors):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            overlay = Image.new("RGBA", image.size, color + (0,))
            overlay_alpha = mask_img.point(lambda v: int(v * alpha))
            overlay.putalpha(overlay_alpha)
            image = Image.alpha_composite(image, overlay)
        return image

    def one_image_inference(self, image_path, prompt, mask_save_path, _id):
        # 读取图像
        image = Image.open(image_path).convert("RGB")
        processor = self.model["processor"]

        with torch.inference_mode(), self._autocast_context():
            inference_state = processor.set_image(image)
            results = processor.set_text_prompt(state=inference_state, prompt=prompt)

        # 保存 mask (0,1 二值图)
        masks = self._masks_to_numpy(results.get("masks"))
        if masks.shape[0] > 0:
            # 如果有多个 mask，直接取最大值合并成单通道
            combined_mask = masks.max(axis=0).astype(np.uint8)
        else:
            combined_mask = np.zeros((image.height, image.width), dtype=np.uint8)

        mask_img = Image.fromarray(combined_mask * 255)
        mask_img.save(mask_save_path)

        # 保存叠加可视化图像
        vis_image = self.overlay_masks(image, masks, alpha=0.5)
        vis_save_path = os.path.join(self.work_dir, f"{_id}_{prompt}_vis.png")
        vis_image.convert("RGB").save(vis_save_path)

        # print(f"[INFO] Saved mask to {mask_save_path}, visualized image to {vis_save_path}")
