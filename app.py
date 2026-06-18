import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
import torch
from transformers import SamModel, SamProcessor


print("Loading SAM model...")
model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
print("Model loaded.")


def apply_mask(image, mask, color=(0, 120, 255), alpha=0.5):
    image_array = np.array(image.convert("RGB"))
    overlay = np.zeros_like(image_array)
    overlay[mask] = color
    result = image_array.copy()
    result[mask] = (
        image_array[mask] * (1 - alpha) + overlay[mask] * alpha
    ).astype(np.uint8)
    return Image.fromarray(result)


def segment(image, evt: gr.SelectData):
    if image is None:
        return None, "Please upload an image first."

    x, y = evt.index
    input_points = [[[x, y]]]

    inputs = processor(image, input_points=input_points, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    scores = outputs.iou_scores.cpu().squeeze()
    best_idx = scores.argmax().item()
    best_mask = masks[0][0][best_idx].numpy()

    result = apply_mask(image, best_mask)

    draw = ImageDraw.Draw(result)
    r = 6
    draw.ellipse(
        [x - r, y - r, x + r, y + r],
        fill="white", outline="black", width=2
    )

    confidence = scores[best_idx].item()
    return result, f"Segmented at ({x}, {y})  |  Confidence: {confidence:.1%}"


with gr.Blocks(title="Zero-Shot Image Segmentation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Zero-Shot Image Segmentation")
    gr.Markdown(
        "Upload any image and **click on any object** to segment it instantly. "
        "Powered by Meta's Segment Anything Model (SAM)."
    )

    with gr.Row():
        input_image = gr.Image(
            label="Click on an object to segment it",
            type="pil"
        )
        output_image = gr.Image(
            label="Segmentation Result",
            type="pil"
        )

    status = gr.Textbox(
        label="Status",
        value="Upload an image and click on any object.",
        interactive=False
    )

    input_image.select(
        fn=segment,
        inputs=input_image,
        outputs=[output_image, status]
    )

demo.launch()