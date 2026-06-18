"""TTS inference entrypoint for the local run.sh wrapper.

Reads inputs from environment variables:
    TEXT                 - text to synthesize
    CFG_VALUE            - classifier-free guidance scale (float)
    INFERENCE_TIMESTEPS  - number of inference steps (int)
    OUTPUT_PATH          - output wav path (default: demo.wav)
    MODEL_DIR            - local model dir (default: ./pretrained_models/VoxCPM2)

Privacy note: all inference runs locally; only the model weights are fetched
from ModelScope on the first run.
"""

import os
from modelscope import snapshot_download
from voxcpm import VoxCPM
import soundfile as sf

text = os.environ["TEXT"]
cfg_value = float(os.environ.get("CFG_VALUE", "2.0"))
inference_timesteps = int(os.environ.get("INFERENCE_TIMESTEPS", "10"))
output_path = os.environ.get("OUTPUT_PATH", "demo.wav")
model_dir = os.environ.get("MODEL_DIR", "./pretrained_models/VoxCPM2")

if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
    print(f"[tts_run] downloading model to {model_dir}", flush=True)
    snapshot_download("OpenBMB/VoxCPM2", local_dir=model_dir)
else:
    print(f"[tts_run] model already present at {model_dir}", flush=True)

print("[tts_run] loading VoxCPM2 ...", flush=True)
model = VoxCPM.from_pretrained(model_dir, load_denoiser=False)

print(f"[tts_run] generating: {text!r}", flush=True)
wav = model.generate(
    text=text,
    cfg_value=cfg_value,
    inference_timesteps=inference_timesteps,
)
sf.write(output_path, wav, model.tts_model.sample_rate)
print(f"[tts_run] wrote {output_path}", flush=True)
