import torch

# SAM3 expects bfloat16 autocast during inference (see examples/sam3_image_predictor_example.ipynb)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

#################################### For Image ####################################
import os

import matplotlib.pyplot as plt
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
# Load the model (weights: sam3.pt in this directory)
CHECKPOINT_PATH = "/root/code/sam3-main/checkpoint/sam3/sam3.pt"
model = build_sam3_image_model(
    checkpoint_path=CHECKPOINT_PATH,
    load_from_HF=False,
)
processor = Sam3Processor(model)
# Load an image
image = Image.open("/root/code/sam3-main/assets/tiger.webp")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="tiger")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# Save visualization to outputs/
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
plot_results(image, output)
save_path = os.path.join(OUTPUT_DIR, "tiger_result.png")
plt.savefig(save_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved visualization to {save_path}")

