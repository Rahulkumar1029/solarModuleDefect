"""Simple training script that downloads Ultralytics' YOLOv11 and fine-tunes it on your dataset.

This script is intentionally small and self-contained. It expects your dataset `data.yaml`
to live at `src/data/files/data.yaml` (the Roboflow-style YAML already present in the repo).

Usage (PowerShell):
    C:/path/to/venv/Scripts/python.exe src/scripts/train.py --epochs 10 --imgsz 1024 --batch 8
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


def ensure_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except Exception:
        print('`ultralytics` not found, attempting to install via pip...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'ultralytics'])
        try:
            from ultralytics import YOLO  # type: ignore

            return YOLO
        except Exception as e:
            print('Failed to import ultralytics after installation:', e)
            raise


def find_data_yaml() -> Path:
    # locate repo root (two parents up from this script: src/scripts -> repo)
    root = Path(__file__).resolve().parents[1]
    candidates = [root / 'src' / 'data' / 'files' / 'data.yaml', root / 'data' / 'files' / 'data.yaml']
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError('data.yaml not found in expected locations: ' + ', '.join(map(str, candidates)))


def main():
    parser = argparse.ArgumentParser(description='Train Ultralytics YOLOv8 on repository dataset')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Ultralytics model spec or weights (e.g. yolov8s.pt)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', type=str, default=None, help="'cpu' or GPU id (e.g. 0) - default: auto")
    args = parser.parse_args()

    data_yaml = find_data_yaml()
    print('Using dataset yaml:', data_yaml)

    YOLO = ensure_ultralytics()

    # decide device
    if args.device is None:
        device_arg = 0 if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu'
    else:
        device_arg = args.device

    print(f'Using model spec: {args.model}')
    print(f'epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={device_arg}')

    # instantiate and run training (Ultralytics will download weights if needed)
    yolo = YOLO(args.model)
    print('Starting Ultralytics training...')
    yolo.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device_arg)
    print('Training finished.')


if __name__ == '__main__':
    main()
