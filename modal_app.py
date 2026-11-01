"""
Run Task 1 (scripts/run_task1.py) on a Modal cloud GPU.

One Modal Volume ("nosh-egg") holds everything that should persist between runs:
  /vol/data/images     the dataset (uploaded once, see README)
  /vol/cache/hf        Hugging Face model weights
  /vol/cache/yolo      YOLO-World weights
  /vol/outputs/<out>   results; download with `modal volume get`

Usage (from the repo root, local venv active):
  python -m modal run modal_app.py --out task1_smoke --limit 3     # smoke test
  python -m modal run modal_app.py --out task1                     # full run
  python -m modal run modal_app.py --out t --methods "gdino_sam3 sam3_text"
"""
import shlex
import subprocess
import sys

import modal

GPU = "H100"
VOL = "/vol"

volume = modal.Volume.from_name("nosh-egg", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    # Same pins as requirements.txt. CLIP is YOLO-World's text encoder; ultralytics would
    # otherwise pip-install it at runtime.
    .pip_install_from_requirements("requirements.txt")
    .env({"HF_HOME": f"{VOL}/cache/hf", "YOLO_CONFIG_DIR": f"{VOL}/cache/yolo"})
    .add_local_dir("scripts", "/root/scripts")
)

app = modal.App("nosh-egg-task1", image=image)


@app.function(gpu=GPU, volumes={VOL: volume}, secrets=[modal.Secret.from_name("huggingface")],
              timeout=3 * 3600)
def task1(out: str, extra_args: list[str]):
    import os
    yolo_dir = f"{VOL}/cache/yolo"
    os.makedirs(yolo_dir, exist_ok=True)
    out_dir = f"{VOL}/outputs/{out}"
    cmd = [sys.executable, "/root/scripts/run_task1.py",
           "--images", f"{VOL}/data/images", "--out", out_dir, *extra_args]
    print(" ".join(cmd), flush=True)
    # cwd = YOLO cache so ultralytics downloads its weights into the volume once.
    try:
        subprocess.run(cmd, check=True, cwd=yolo_dir)
    finally:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/pip_freeze.txt", "w") as f:
            f.write(freeze.stdout)
        volume.commit()


@app.local_entrypoint()
def main(out: str = "task1", limit: int = 0, methods: str = "", extra: str = ""):
    args = []
    if limit:
        args += ["--limit", str(limit)]
    if methods:
        args += ["--methods", *methods.split()]
    args += shlex.split(extra)
    task1.remote(out, args)
    print(f"\nDone. Download with:\n  python -m modal volume get nosh-egg /outputs/{out} outputs/")
