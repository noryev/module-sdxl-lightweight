import os
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
import numpy as np
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Print all environment variables
logging.info("All Environment Variables:")
for key, value in os.environ.items():
    logging.info(f"{key}: {value}")

# Explicitly log the PROMPT environment variable
prompt = os.getenv("PROMPT")
logging.info(f"PROMPT environment variable: {prompt}")

# Use the prompt (rest of your script)
prompt = os.getenv("PROMPT", "A lilypad floating on a galaxy of stars")
logging.info(f"Using prompt: {prompt}")

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

# Log all environment variables
logging.info("Environment variables:")
for key, value in os.environ.items():
    logging.info(f"{key}: {value}")

seed = int(os.getenv("SEED", "42"))
logging.info(f"Using seed: {seed}")
set_seed(seed)

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
g.manual_seed(seed)

prompt = os.getenv("PROMPT", "A lilypad floating on a galaxy of stars")
logging.info(f"Using prompt: {prompt}")

num_inference_steps = int(os.getenv("STEPS", "30"))
height = int(os.getenv("HEIGHT", "512"))
width = int(os.getenv("WIDTH", "512"))
logging.info(f"Inference steps: {num_inference_steps}, Height: {height}, Width: {width}")

with torch.inference_mode():
    images = pipe(
        prompt=prompt,
        generator=g,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
    ).images

logging.info(f"Generated {len(images)} images")

image = images[0]

output_dir = os.getenv("OUTPUT_DIR", "/outputs/")
logging.info(f"Output directory: {output_dir}")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"image-{seed}.png")
image.save(output_path)
logging.info(f"Image saved as {output_path}")

# Create a done file to signal job completion
done_file = os.path.join(output_dir, "DONE")
with open(done_file, "w") as f:
    f.write("Job completed successfully")
logging.info(f"Created DONE file: {done_file}")

logging.info("Script completed successfully")