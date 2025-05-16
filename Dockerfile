FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg libgl1 wget unzip python3-pip

# Set working directory
WORKDIR /workspace

# Clone your repo
RUN git clone https://github.com/arpitdayma3/LatentSyncnew.git  .

# Install PyTorch compatible with CUDA 12.9
RUN pip install --break-system-packages \
    torch==2.5.1+cu129 \
    torchvision==0.20.1+cu129 \
    torchaudio==2.5.1 \
    --extra-index-url https://download.pytorch.org/whl/cu129 

# Install requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Setup checkpoints and handler logic
COPY setup_checkpoints.sh .
RUN chmod +x setup_checkpoints.sh && ./setup_checkpoints.sh

COPY handler.py .
EXPOSE 8000
