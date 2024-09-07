import os
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
import numpy as np
import random
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up argument parsing
parser = argparse.ArgumentParser(description='Run SDXL with custom parameters')
parser.add_argument('--prompt', type=str, default="A lilypad floating on a galaxy of stars", help='The prompt for image generation')
parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
parser.add_argument('--steps', type=int, default=30, help='Number of inference steps')
parser.add_argument('--height', type=int, default=512, help='Image height')
parser.add_argument('--width', type=int, default=512, help='Image width')
parser.add_argument('--output_dir', type=str, default="/outputs/", help='Output directory for generated images')
args = parser.parse_args()

logging.info(f"Received arguments: {args}")

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")

set_seed(args.seed)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

g = torch.Generator(device="cuda")
g.manual_seed(args.seed)

logging.info(f"Using prompt: {args.prompt}")
logging.info(f"Inference steps: {args.steps}, Height: {args.height}, Width: {args.width}")

with torch.inference_mode():
    images = pipe(
        prompt=args.prompt,
        generator=g,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
    ).images

logging.info(f"Generated {len(images)} images")

image = images[0]

os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, f"image-{args.seed}.png")
image.save(output_path)
logging.info(f"Image saved as {output_path}")

# Create a done file to signal job completion
done_file = os.path.join(args.output_dir, "DONE")
with open(done_file, "w") as f:
    f.write("Job completed successfully")
logging.info(f"Created DONE file: {done_file}")

logging.info("Script completed successfully")