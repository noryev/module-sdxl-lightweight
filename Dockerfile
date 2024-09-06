# Use a more recent CUDA base image
FROM python:3.10-slim-bullseye

ARG HUGGINGFACE_TOKEN

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir huggingface_hub==0.16.4

# Login to Hugging Face and download the model
RUN python3 -c "from huggingface_hub import login; login('${HUGGINGFACE_TOKEN}')" \
    && python3 -c "from diffusers import DiffusionPipeline; import torch; DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-0.9', torch_dtype=torch.float16, use_safetensors=True, variant='fp16')" \
    && rm -f ~/.huggingface/token

# Copy the inference script
COPY inference.py .

# Set the entrypoint
ENTRYPOINT ["python3", "/app/inference.py"]