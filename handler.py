import os
import subprocess
import uuid
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    video_path: str
    audio_path: str
    guidance_scale: float = 2.0
    inference_steps: int = 20
    seed: int = 1247

@app.post("/run")
def run_pipeline(data: InputData):
    output_dir = f"/workspace/outputs/{uuid.uuid4()}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = f"""
    cd /workspace &&
    python -m scripts.inference \
        --unet_config_path configs/unet/stage2_efficient.yaml \
        --inference_ckpt_path checkpoints/latentsync_unet.pt \
        --guidance_scale {data.guidance_scale} \
        --video_path {data.video_path} \
        --audio_path {data.audio_path} \
        --video_out_path {output_dir}/output.mp4 \
        --seed {data.seed} \
        --inference_steps {data.inference_steps}
    """
    try:
        subprocess.run(cmd, shell=True, check=True)
        return {"output": f"{output_dir}/output.mp4"}
    except Exception as e:
        return {"error": str(e)}
