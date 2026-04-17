#!/usr/bin/env python3
"""AutoEncoder training and feature-vector export for behavior clustering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_KEY = "image"
CHANNEL_ORDER = ("aerial", "layout", "resist", "epe", "pv", "nils")


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Manifest row must be an object at {path}:{line_no}")
            records.append(item)
    if not records:
        raise ValueError("Manifest is empty")
    return records


def _resolve_path(path_text: str, base_dir: Path) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _load_npz_image(path: str | Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as data:
        if IMAGE_KEY not in data:
            raise ValueError(f"NPZ {path} must contain key '{IMAGE_KEY}'")
        image = np.asarray(data[IMAGE_KEY], dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2:
        raise ValueError(f"Image in {path} must be 2-D")
    return np.ascontiguousarray(image, dtype=np.float32)


def _manifest_channels(records: Sequence[Dict[str, Any]]) -> List[str]:
    channels = ["aerial"]
    for channel in CHANNEL_ORDER[1:]:
        key = f"{channel}_npz"
        present = [idx for idx, record in enumerate(records) if record.get(key)]
        if present and len(present) != len(records):
            raise ValueError(f"Optional channel {channel} must be present for all rows or none")
        if present:
            channels.append(channel)
    return channels


def _sample_paths(record: Dict[str, Any], base_dir: Path, channels: Sequence[str]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for channel in channels:
        key = f"{channel}_npz"
        if key not in record:
            raise ValueError(f"Manifest row missing {key}")
        paths[channel] = _resolve_path(str(record[key]), base_dir)
    return paths


def _load_stack(paths: Dict[str, str], channels: Sequence[str]) -> np.ndarray:
    images = [_load_npz_image(paths[channel]) for channel in channels]
    shape = images[0].shape
    for image in images:
        if image.shape != shape:
            raise ValueError(f"All images for one sample must share shape; got {image.shape} vs {shape}")
    return np.stack(images, axis=0).astype(np.float32)


class _ImageManifestDataset:
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.resolve().parent
        self.records = _read_jsonl(self.manifest_path)
        self.channels = _manifest_channels(self.records)
        self.sample_ids = [str(record["sample_id"]) for record in self.records]
        self.paths = [_sample_paths(record, self.base_dir, self.channels) for record in self.records]
        first = _load_stack(self.paths[0], self.channels)
        self.shape = tuple(int(value) for value in first.shape)
        for paths in self.paths[1:]:
            if _load_stack(paths, self.channels).shape != first.shape:
                raise ValueError("All manifest images must share the same channel/height/width shape")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        stack = _load_stack(self.paths[int(index)], self.channels)
        return stack, self.sample_ids[int(index)]


def _make_torch_dataset(base_dataset):
    class TorchDataset(Dataset):
        def __len__(self):
            return len(base_dataset)

        def __getitem__(self, index):
            stack, sample_id = base_dataset[index]
            return torch.from_numpy(stack), sample_id

    return TorchDataset()


def _make_model_class():
    class ConvAutoEncoder(nn.Module):
        def __init__(self, channels: int, height: int, width: int, latent_dim: int):
            super().__init__()
            self.channels = int(channels)
            self.height = int(height)
            self.width = int(width)
            self.latent_dim = int(latent_dim)
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            self.encoder_fc = nn.Linear(64 * 4 * 4, self.latent_dim)
            self.decoder_fc = nn.Linear(self.latent_dim, 64 * 4 * 4)
            self.decoder_conv = nn.Sequential(
                nn.Upsample(size=(self.height, self.width), mode="bilinear", align_corners=False),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, self.channels, kernel_size=3, padding=1),
            )

        def encode(self, x):
            return self.encoder_fc(self.encoder_conv(x))

        def decode(self, z):
            x = self.decoder_fc(z).view(-1, 64, 4, 4)
            return self.decoder_conv(x)

        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z

    return ConvAutoEncoder


def _global_ssim_loss(pred, target):
    x = pred[:, :1]
    y = target[:, :1]
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = x.mean(dim=(-2, -1), keepdim=True)
    mu_y = y.mean(dim=(-2, -1), keepdim=True)
    var_x = ((x - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
    var_y = ((y - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
    cov = ((x - mu_x) * (y - mu_y)).mean(dim=(-2, -1), keepdim=True)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return torch.clamp(1.0 - ssim, 0.0, 2.0).mean()


def train(args: argparse.Namespace) -> int:
    base_dataset = _ImageManifestDataset(args.manifest)
    dataset = _make_torch_dataset(base_dataset)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True)
    channels, height, width = base_dataset.shape
    model_cls = _make_model_class()
    model = model_cls(channels, height, width, int(args.latent_dim))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    mse_loss = nn.MSELoss()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for batch, _ in loader:
            batch = batch.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            recon, _ = model(batch)
            loss = mse_loss(recon, batch) + float(args.ssim_weight) * _global_ssim_loss(recon, batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * int(batch.shape[0])
            count += int(batch.shape[0])
        print(f"epoch {epoch:04d}: loss={total / max(count, 1):.6f}")

    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "channels": int(channels),
            "height": int(height),
            "width": int(width),
            "latent_dim": int(args.latent_dim),
            "channel_names": list(base_dataset.channels),
        },
        output,
    )
    print(f"model saved to: {output}")
    return 0


def encode(args: argparse.Namespace) -> int:
    checkpoint = torch.load(args.model, map_location="cpu")
    base_dataset = _ImageManifestDataset(args.manifest)
    channels, height, width = base_dataset.shape
    expected = (int(checkpoint["channels"]), int(checkpoint["height"]), int(checkpoint["width"]))
    if (channels, height, width) != expected:
        raise ValueError(f"Manifest image shape {(channels, height, width)} does not match model shape {expected}")
    model_cls = _make_model_class()
    model = model_cls(channels, height, width, int(checkpoint["latent_dim"]))
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()

    loader = DataLoader(_make_torch_dataset(base_dataset), batch_size=int(args.batch_size), shuffle=False)
    all_ids: List[str] = []
    all_features: List[np.ndarray] = []
    with torch.no_grad():
        for batch, sample_ids in loader:
            batch = batch.to(device=device, dtype=torch.float32)
            features = model.encode(batch).detach().cpu().numpy().astype(np.float32)
            all_features.append(features)
            all_ids.extend([str(value) for value in sample_ids])
    features_np = np.vstack(all_features).astype(np.float32)

    feature_path = Path(args.features_out)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_path, sample_ids=np.asarray(all_ids, dtype=str), features=features_np)

    fv_manifest = Path(args.fv_manifest_out)
    fv_manifest.parent.mkdir(parents=True, exist_ok=True)
    with fv_manifest.open("w", encoding="utf-8") as handle:
        for sample_id in all_ids:
            handle.write(json.dumps({"sample_id": sample_id, "feature_npz": str(feature_path)}, ensure_ascii=False) + "\n")
    print(f"features saved to: {feature_path}")
    print(f"fv manifest saved to: {fv_manifest}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/encode layout behavior AutoEncoder feature vectors")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train AutoEncoder")
    train_parser.add_argument("--manifest", required=True, help="Training JSONL manifest")
    train_parser.add_argument("--model-out", required=True, help="Output .pt model path")
    train_parser.add_argument("--latent-dim", type=int, default=128, help="Feature vector dimension")
    train_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    train_parser.add_argument("--ssim-weight", type=float, default=1.0, help="Aerial SSIM loss weight")
    train_parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu")
    train_parser.set_defaults(func=train)

    encode_parser = subparsers.add_parser("encode", help="Encode manifest into feature vectors")
    encode_parser.add_argument("--manifest", required=True, help="JSONL manifest to encode")
    encode_parser.add_argument("--model", required=True, help="Input .pt model path")
    encode_parser.add_argument("--features-out", required=True, help="Output features.npz path")
    encode_parser.add_argument("--fv-manifest-out", required=True, help="Output FV manifest JSONL path")
    encode_parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    encode_parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu")
    encode_parser.set_defaults(func=encode)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"运行失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
