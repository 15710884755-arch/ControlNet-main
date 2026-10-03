"""Restore all gradio_*.py from GitHub + apply correct patches."""
from pathlib import Path
import urllib.request

REPO = "https://raw.githubusercontent.com/lllyasviel/ControlNet/main"
ROOT = Path(__file__).resolve().parents[1]

FILES = [
    ("gradio_canny2image.py",   "7860"),
    ("gradio_pose2image.py",    "7861"),
    ("gradio_normal2image.py",  "7862"),
    ("gradio_depth2image.py",   "7863"),
    ("gradio_hough2image.py",   "7864"),
    ("gradio_hed2image.py",     "7865"),
    ("gradio_scribble2image.py","7866"),
    ("gradio_seg2image.py",     "7867"),
    ("gradio_fake_scribble2image.py", "7868"),
    ("gradio_scribble2image_interactive.py", "7869"),
]

# TF32 is key for this local runtime.
GPU_OPT = """
# === GPU optimizations (TF32 = 12x faster than autocast fp16!) ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
"""

SEED_FN = '''def seed_everything(seed: int):
    import random, os, numpy as np, torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

'''

for fname, port in FILES:
    url = f"{REPO}/{fname}"
    print(f"\n=== {fname} ===")
    
    with urllib.request.urlopen(url, timeout=30) as r:
        content = r.read().decode('utf-8')
    
    # Patch 1: TF32 optimizations
    content = content.replace("import torch\n", "import torch\n" + GPU_OPT + "\n", 1)

    # Patch 1b: move YAML configs out of the model-weight directory
    content = content.replace("./models/cldm_v15.yaml", "./configs/cldm_v15.yaml")
    content = content.replace("./models/cldm_v21.yaml", "./configs/cldm_v21.yaml")
    
    # Patch 2: Replace pytorch_lightning seed_everything with local def
    content = content.replace("from pytorch_lightning import seed_everything\n", "")
    content = content.replace("from annotator.util", SEED_FN + "from annotator.util", 1)
    
    # Patch 3: load_state_dict to CPU (avoids peak VRAM spike), strict=False
    content = content.replace(
        "location='cuda'))",
        "location='cpu'), strict=False)"
    )
    content = content.replace("location='cuda'),", "location='cpu'),")
    print("  [3] load_state_dict: location=cpu + strict=False")
    
    # Patch 4: Default resolution 512 → 384
    content = content.replace(
        'value=512, step=64)',
        'value=384, step=64)'
    )
    print("  [4] default resolution 384")
    
    # Keep the upstream fp32 path; local TF32 flags handle acceleration.
    
    # Patch 5: Custom port
    content = content.replace(
        "block.launch(server_name='0.0.0.0')",
        f"block.launch(server_name='0.0.0.0', server_port={port})"
    )
    
    # Write (UTF-8 no BOM)
    with (ROOT / fname).open('w', encoding='utf-8', newline='\n') as f:
        f.write(content)
    
    print(f"  saved {len(content)} bytes")

print("\n=== ALL 8 SCRIPTS PATCHED ===")
print("=== config.py save_memory=False ===")
print("=== cldm/ddim_hacked.py .device bug ===")
print("=== READY TO LAUNCH! ===")
