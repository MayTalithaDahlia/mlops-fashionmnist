import hashlib
import json
from pathlib import Path
from torchvision.datasets import FashionMNIST

ROOT = Path("data")
RAW = ROOT / "FashionMNIST"
ROOT.mkdir(exist_ok=True)

def main():
    # Unduh dataset resmi Fashion-MNIST ke folder proyek.
    FashionMNIST(root=str(ROOT), train=True, download=True)
    FashionMNIST(root=str(ROOT), train=False, download=True)

    digest = hashlib.sha256()
    for path in sorted(RAW.rglob("*")):
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    data_hash = digest.hexdigest()
    metadata = {
        "dataset": "Fashion-MNIST",
        "sha256": data_hash,
        "train_samples": 60000,
        "test_samples": 10000,
        "classes": 10
    }
    (ROOT / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print("Dataset siap. Catat hash ini pada setiap run MLflow.")

if __name__ == "__main__":
    main()
