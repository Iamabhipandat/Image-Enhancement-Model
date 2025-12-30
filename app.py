import gradio as gr
import torch 
import kornia as K
import numpy as np
from PIL import Image


def equalize_image(image: Image.Image) -> Image.Image:
    img_np = np.array(image).astype(np.float32) / 255.0

    img_tensor = (
        torch.from_numpy(img_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    out_tensor = K.enhance.equalize(img_tensor)

    out_np = (
        out_tensor.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    out_np = (out_np * 255).astype(np.uint8)

    return Image.fromarray(out_np)


demo = gr.Interface(
    fn=equalize_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(type="pil", label="Equalized Image"),
    title="Kornia Image Equalization Demo",
    description="Upload an image to see the effect of histogram equalization using Kornia."
)


if __name__ == "__main__":
    demo.launch()
