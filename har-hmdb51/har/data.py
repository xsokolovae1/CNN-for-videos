"""Načítanie snímok klipu a ich príprava ako vstup pre neurónové siete."""
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import N_FRAMES, SIZE

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def even_indices(length, n):
    return np.linspace(0, length - 1, n).round().astype(int)


def resize_frames(frames, size=SIZE):
    return np.stack([cv2.resize(f, (size, size)) for f in frames])


def sample_frames(frames, n=N_FRAMES):
    # RGB snímok  yberie n rovnomerne rozložených 
    return resize_frames([frames[i] for i in even_indices(len(frames), n)])


def read_frames(path, n=N_FRAMES):
    files = sorted(Path(path).glob("*.jpg"), key=lambda f: int(f.stem))
    if not files:
        raise RuntimeError(f"V priečinku {path} nie sú žiadne snímky")
    frames = [cv2.cvtColor(cv2.imread(str(files[i])), cv2.COLOR_BGR2RGB)
              for i in even_indices(len(files), n)]
    return resize_frames(frames)


def to_imagenet_tensor(frames):
    #pre ResNet18.
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def to_video_tensor(frames, video_transform):
    #pre video sieť R(2+1)D
    x = torch.from_numpy(frames).permute(0, 3, 1, 2)
    return video_transform(x).unsqueeze(0)
