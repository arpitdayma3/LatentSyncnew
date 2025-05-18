import os
import time
import requests
import subprocess

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/chunyu-li/LatentSync/model.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def upload_file_to_transfer_sh(file_path):
    with open(file_path, 'rb') as f:
        response = requests.put(f"https://transfer.sh/{os.path.basename(file_path)}", data=f)
    return response.text.strip()


def main(inputs):
    # Get inputs from request
    video_url = inputs["video_url"]
    audio_url = inputs["audio_url"]
    guidance_scale = float(inputs.get("guidance_scale", 2.0))
    inference_steps = int(inputs.get("inference_steps", 20))
    seed = int(inputs.get("seed", 0))

    if seed <= 0:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    # Prepare input file paths
    video_path = "/tmp/input_video.mp4"
    audio_path = "/tmp/input_audio.wav"
    output_path = "/tmp/output_video.mp4"

    # Download input files
    print("Downloading input files...")
    download_file(video_url, video_path)
    download_file(audio_url, audio_path)

    # Define paths
    config_path = "configs/unet/stage2.yaml"
    ckpt_path = "checkpoints/latentsync_unet.pt"

    # Run inference command
    print("Running inference...")
    os.system(
        f"python -m scripts.inference "
        f"--unet_config_path {config_path} "
        f"--inference_ckpt_path {ckpt_path} "
        f"--guidance_scale {guidance_scale} "
        f"--video_path {video_path} "
        f"--audio_path {audio_path} "
        f"--video_out_path {output_path} "
        f"--seed {seed} "
        f"--inference_steps {inference_steps}"
    )

    print("Uploading output...")
    result_url = upload_file_to_transfer_sh(output_path)

    return result_url
