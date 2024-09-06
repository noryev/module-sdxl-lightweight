import os
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
import numpy as np
import random

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
    print(f"Random seed set as {seed}")

seed = int(os.getenv("SEED", "42"))
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

prompt = os.getenv("PROMPT", "An astronaut riding a green horse")

with torch.inference_mode():
    images = pipe(
        prompt=prompt,
        generator=g,
        num_inference_steps=30,
        height=512,
        width=512,
    ).images

print(f"Got {len(images)} images")

image = images[0]

output_dir = os.getenv("OUTPUT_DIR", "/outputs/")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"image-{seed}.png")
image.save(output_path)
print(f"Image saved as {output_path}")

# Create a done file to signal job completion
with open(os.path.join(output_dir, "DONE"), "w") as f:
    f.write("Job completed successfully")