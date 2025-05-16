FROM nvidia/cuda:12.4.0.base

# Install system packages
RUN apt-get update && apt-get install -y git ffmpeg libgl1 wget unzip python3-pip

# Set working directory
WORKDIR /workspace

# Clone your forked repo
RUN git clone https://github.com/arpitdayma3/LatentSyncnew.git  .
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download checkpoints
COPY setup_checkpoints.sh .
RUN chmod +x setup_checkpoints.sh && ./setup_checkpoints.sh

# Copy handler
COPY handler.py .

# Expose port
EXPOSE 8000
