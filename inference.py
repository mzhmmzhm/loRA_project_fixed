# infer.py

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def run_inference(
    model_id: str,
    lora_path: str,
    prompt: str,
    negative_prompt: str = None,
    guidance_scale: float = 7.5
) -> Image.Image:
    """
    Generates an image using a Stable Diffusion pipeline fine-tuned with LoRA weights.

    Args:
        model_id (str): Hugging Face Hub model identifier for the base model. #The trained LoRA in this repo is for SD 2.1.
        lora_path (str): Path to the directory containing saved LoRA weights.
        prompt (str): Text prompt for image generation.
        negative_prompt (str, optional): Text prompt for features to avoid (default: None).
        guidance_scale (float, optional): How strongly the model should follow the prompt (default: 7.5).

    Returns:
        PIL.Image.Image: Generated image.
    """
    # 1. Load the base Stable Diffusion pipeline (UNet, VAE, Text Encoder).
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")

    # 2. Load the fine-tuned LoRA weights into the pipeline.
    print(f"Loading LoRA weights from {lora_path}...")
    pipeline.load_lora_weights(lora_path)

    # 3. Generate the image.
    print(f"Generating image for prompt:\n  \"{prompt}\"")
    with torch.no_grad():
        output = pipeline(
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale
        )
        image = output.images[0]

    print("Image generation complete.")
    return image
