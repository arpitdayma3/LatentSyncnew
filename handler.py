import runpod
import os
from predict import main as run_latentsync


# Ensure checkpoints and cache directories exist
def setup_environment():
    os.makedirs("checkpoints/whisper", exist_ok=True)
    os.makedirs("checkpoints/auxiliary", exist_ok=True)
    os.makedirs(os.path.expanduser("~/.cache/torch/hub/checkpoints"), exist_ok=True)

    # Create symlinks for auxiliary models (if not already)
    aux_files = [
        "2DFAN4-cd938726ad.zip",
        "s3fd-619a316812.pth",
        "vgg16-397923af.pth",
    ]
    for filename in aux_files:
        src = os.path.abspath(f"checkpoints/auxiliary/{filename}")
        dst = os.path.expanduser(f"~/.cache/torch/hub/checkpoints/{filename}")
        if not os.path.exists(dst) and os.path.exists(src):
            os.symlink(src, dst)


# Run once on cold start
setup_environment()


def handler(job):
    try:
        inputs = job["input"]

        # Required fields: video_url and audio_url
        if not inputs.get("video_url") or not inputs.get("audio_url"):
            return {"status": "error", "message": "Missing video_url or audio_url."}

        print("Running LatentSync inference job...")

        # Call the model
        result_url = run_latentsync(inputs)

        return {
            "status": "success",
            "output_url": result_url
        }

    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"status": "error", "message": str(e)}


runpod.serverless.start({"handler": handler})
