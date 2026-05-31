from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = [
    "google-bert/bert-base-uncased",
    "IDEA-Research/grounding-dino-base",
    "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-base-plus",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
]

ROOT = Path("models")
ROOT.mkdir(exist_ok=True)

for repo_id in MODELS:
    model_name = repo_id.split("/")[-1]
    target_dir = ROOT / model_name

    print(f"\nDownloading {repo_id}")
    print(f"Target: {target_dir}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Finished: {repo_id}")

print("\nAll models downloaded successfully.")
