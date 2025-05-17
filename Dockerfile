# Use an official Python runtime with CUDA support as a parent image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the model checkpoints
RUN huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir checkpoints && \
    huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints

# Copy the rest of the application code
COPY . .

# Create a new Conda environment
RUN conda create -y -n latentsync python=3.10.13 && \
    conda activate latentsync && \
    conda install -y -c conda-forge ffmpeg

# Set the entry point for the container
ENTRYPOINT ["python", "-m", "scripts.inference"]
