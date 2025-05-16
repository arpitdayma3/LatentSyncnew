# Use a supported CUDA + Ubuntu base image
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# Set working directory
WORKDIR /workspace

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 wget unzip python3-pip python3-venv \
    build-essential libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv && \
    echo "source /opt/venv/bin/activate" >> ~/.bashrc

ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch (CUDA 12.1 compatible)
RUN pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121 

# Clone the repo
RUN git clone https://github.com/arpitdayma3/LatentSyncnew.git  .

# Install requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download checkpoints
COPY setup_checkpoints.sh .
RUN chmod +x setup_checkpoints.sh && ./setup_checkpoints.sh

# Copy handler
COPY handler.py .

# Expose FastAPI port
EXPOSE 8000

# Optional: Run Gradio app by default
CMD ["python", "gradio_app.py"]
