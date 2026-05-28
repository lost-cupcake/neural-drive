"""
Entry point: python train.py --config configs/nuscenes.yaml
             python train.py --config configs/nuscenes.yaml --resume checkpoints/epoch_005.pth
"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from training.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def main():
    parser = argparse.ArgumentParser(description='MTL Autonomous Driving Training')
    parser.add_argument('--config', default='configs/nuscenes.yaml', help='Config YAML path')
    parser.add_argument('--resume', default=None, help='Checkpoint path to resume from')
    args = parser.parse_args()

    # ── Reproducibility ────────────────────────────────────────────────
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg['system'].get('seed', 42))

    # ── Build and run trainer ──────────────────────────────────────────
    trainer = Trainer(args.config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    print("\n" + "=" * 60)
    print(" Multi-Task Learning — Autonomous Driving (LiDAR + Camera)")
    print("=" * 60)
    print(f" Config:  {args.config}")
    print(f" Resume:  {args.resume or 'No'}")
    print(f" Device:  {trainer.device}")
    print("=" * 60 + "\n")

    trainer.train()


if __name__ == '__main__':
    main()
