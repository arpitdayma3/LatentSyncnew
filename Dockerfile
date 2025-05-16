FROM nvidia/cuda:12.1.0-devel-ubuntu20.04 

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 wget unzip \
    python3-pip python3-venv build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv && \
    echo "source /opt/venv/bin/activate" >> ~/.bashrc

ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch (CUDA 12.9)
RUN pip install torch==2.5.1+cu129 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 

# Clone repo and proceed with setup
RUN git clone https://github.com/arpitdayma3/LatentSyncnew.git .

# Copy files like requirements.txt, handler.py, etc.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Setup checkpoints
COPY setup_checkpoints.sh .
RUN chmod +x setup_checkpoints.sh && ./setup_checkpoints.sh

# Expose FastAPI port
EXPOSE 8000
