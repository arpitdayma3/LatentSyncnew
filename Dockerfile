FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 wget unzip \
    python3-pip python3-venv build-essential \
    python3-dev g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv && \
    echo "source /opt/venv/bin/activate" >> ~/.bashrc

ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.4 support
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu124 

# Optional: Install Cython early for compiling packages like insightface
RUN pip install cython

# Clone repo and proceed with setup
RUN git clone https://github.com/arpitdayma3/LatentSyncnew.git  .

# Copy files like requirements.txt, handler.py, etc.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Setup checkpoints
COPY setup_checkpoints.sh .
RUN chmod +x setup_checkpoints.sh && ./setup_checkpoints.sh

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
