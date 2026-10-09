import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from mlflow import MlflowClient

DATA_DIR = Path("data")
CONFIG_PATH = Path("best_config.json")
EXPERIMENT_NAME = "fashion-mnist-cnn"
MODEL_NAME = "fashion-mnist-cnn"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data(batch_size=128, augment=False, seed=42):
    # Augmentasi hanya diterapkan pada data training.
    train_transform = [transforms.ToTensor(),
                       transforms.Normalize((0.2860,), (0.3530,))]
    if augment:
        train_transform.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    train_ds = datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=True,
        transform=transforms.Compose(train_transform)
    )
    val_ds = datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))
        ])
    )
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(len(train_ds)), [55000, 5000], generator=generator
    )
    train_subset = Subset(train_ds, train_indices.indices)
    # Validasi memakai transform tanpa augmentasi agar evaluasi konsisten.
    val_subset = Subset(val_ds, val_indices.indices)
    # Test set tidak dipakai selama tuning.
    test_ds = datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))
        ])
    )
    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              generator=loader_gen, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0)
    return train_loader, val_loader, test_loader

def data_hash():
    meta = DATA_DIR / "dataset_metadata.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))["sha256"]
    # Fallback: hash metadata yang stabil; jalankan prepare_data.py untuk hash file sebenarnya.
    return hashlib.sha256(b"Fashion-MNIST;train=60000;test=10000").hexdigest()

class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.25, channels=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 2 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total

def train_model(config, epochs, seed=42, augment=False, log_model=True):
    set_seed(seed)
    batch_size = int(config.get("batch_size", 128))
    train_loader, val_loader, _ = get_data(batch_size, augment=augment, seed=seed)
    model = SimpleCNN(
        dropout=float(config.get("dropout", 0.25)),
        channels=int(config.get("channels", 16))
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config.get("lr", 0.001))
    )
    best_val_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }, step=epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    mlflow.log_metric("best_val_accuracy", best_val_acc)
    
    if log_model:
        input_example = torch.randn(1, 1, 28, 28)

        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=input_example,
            serialization_format="pickle"
        )

    return model, best_val_acc

def setup():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def log_common(seed, config, augment):
    mlflow.log_params({k: v for k, v in config.items()})
    mlflow.log_param("seed", seed)
    mlflow.log_param("device", str(DEVICE))
    mlflow.log_param("augmentation", str(augment))
    mlflow.set_tag("dataset", "Fashion-MNIST")
    mlflow.set_tag("dataset_sha256", data_hash())

def baseline(args):
    config = {"lr": 0.001, "batch_size": 128, "dropout": 0.25, "channels": 16}
    with mlflow.start_run(run_name="baseline-cnn"):
        log_common(42, config, False)
        mlflow.set_tag("stage", "baseline")
        train_model(config, args.epochs, seed=42, augment=False)

def tune(args):
    # Parent run mewakili sweep; setiap trial menjadi nested child run.
    with mlflow.start_run(run_name="optuna-sweep"):
        mlflow.log_param("n_trials", args.trials)
        mlflow.log_param("epochs_per_trial", args.epochs)
        def objective(trial):
            config = {
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "channels": trial.suggest_categorical("channels", [16, 32])
            }
            with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
                log_common(42, config, False)
                mlflow.set_tag("stage", "tuning")
                mlflow.log_param("trial_number", trial.number)
                _, score = train_model(config, args.epochs, seed=42, augment=False)
                mlflow.log_metric("objective_val_accuracy", score)
                return score
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=args.trials)
        CONFIG_PATH.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
        mlflow.log_dict(study.best_params, "best_config.json")
        mlflow.log_metric("best_trial_val_accuracy", study.best_value)
        print("Konfigurasi terbaik:", study.best_params)
        print("Akurasi validasi terbaik:", study.best_value)
        print("Disimpan ke", CONFIG_PATH)

def load_best_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Jalankan mode tune terlebih dahulu untuk membuat best_config.json")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

def evaluate_seeds(args):
    config = load_best_config()
    scores = []
    for seed in [42, 123, 2026]:
        with mlflow.start_run(run_name=f"best-config-seed-{seed}"):
            log_common(seed, config, False)
            mlflow.set_tag("stage", "multi-seed-evaluation")
            _, score = train_model(config, args.epochs, seed=seed, augment=False)
            scores.append(score)
    mean, std = float(np.mean(scores)), float(np.std(scores, ddof=1))
    with mlflow.start_run(run_name="three-seed-summary"):
        mlflow.log_params(config)
        mlflow.log_metric("val_accuracy_mean", mean)
        mlflow.log_metric("val_accuracy_std", std)
        mlflow.log_dict({"seeds": [42, 123, 2026], "scores": scores,
                         "mean": mean, "std_sample": std},
                        "three_seed_results.json")
    print("Skor per seed:", scores)
    print(f"Mean akurasi validasi = {mean:.4f}")
    print(f"Standar deviasi sampel = {std:.4f}")

def ablation(args):
    config = load_best_config()
    variants = [
        ("full-model", True, float(config.get("dropout", 0.25))),
        ("without-augmentation", False, float(config.get("dropout", 0.25))),
        ("without-dropout", True, 0.0)
    ]
    for name, augment, dropout in variants:
        variant_config = {**config, "dropout": dropout}
        with mlflow.start_run(run_name=f"ablation-{name}"):
            log_common(42, variant_config, augment)
            mlflow.set_tag("stage", "ablation")
            mlflow.set_tag("ablation_variant", name)
            _, score = train_model(variant_config, args.epochs, seed=42,
                                   augment=augment)
            mlflow.log_metric("ablation_val_accuracy", score)
            print(name, "validation accuracy:", round(score, 4))

def register(args):
    client = MlflowClient(tracking_uri=TRACKING_URI)
    # Pilih run tuning dengan nilai objective validasi tertinggi.
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="tags.stage = 'tuning'",
        order_by=["metrics.objective_val_accuracy DESC"],
        max_results=1
    )
    if runs.empty:
        raise RuntimeError("Run tuning tidak ditemukan. Jalankan mode tune dahulu.")
    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri, MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", registered.version)
    loaded = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}@champion")
    _, _, test_loader = get_data(batch_size=128, augment=False, seed=42)
    loaded.to(DEVICE)
    test_loss, test_acc = run_epoch(loaded, test_loader, nn.CrossEntropyLoss())
    with mlflow.start_run(run_name="champion-test-evaluation"):
        mlflow.log_param("source_run_id", run_id)
        mlflow.log_param("registered_model_version", registered.version)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.set_tag("model_alias", "champion")
        mlflow.set_tag("dataset_sha256", data_hash())
    sample = torch.randn(2, 1, 28, 28).to(DEVICE)
    loaded.eval()
    with torch.no_grad():
        predictions = loaded(sample).argmax(dim=1).cpu().tolist()
    print("Model terdaftar:", MODEL_NAME, "versi", registered.version)
    print("Alias: @champion")
    print("Test accuracy:", round(test_acc, 4))
    print("Contoh prediksi kelas (input dummy):", predictions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "tune", "seeds", "ablation", "register"],
                        required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    setup()
    if args.mode == "baseline":
        baseline(args)
    elif args.mode == "tune":
        if args.trials < 20:
            raise ValueError("Tugas meminta minimal 20 trial; gunakan --trials 20 atau lebih.")
        tune(args)
    elif args.mode == "seeds":
        evaluate_seeds(args)
    elif args.mode == "ablation":
        ablation(args)
    elif args.mode == "register":
        register(args)

if __name__ == "__main__":
    main()
