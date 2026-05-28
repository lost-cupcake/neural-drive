"""
Trainer: full training loop with:
  - AMP (mixed precision)
  - TensorBoard logging
  - Gradient clipping
  - Cosine LR schedule with warmup
  - Checkpointing
"""

import os
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mtl_model      import MTLAutonomousModel, MTLConfig
from training.losses        import MTLLoss
from datasets.nuscenes_mtl import NuScenesMTL, collate_fn


class Trainer:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = self.cfg['training'].get('amp', True) and self.device.type == 'cuda'
        print(f"[Trainer] Device: {self.device} | AMP: {self.use_amp}")

        self._setup_dirs()
        self._build_model()
        self._build_datasets()
        self._build_optimizer()
        self._build_loss()

        self.writer  = SummaryWriter(self.log_dir)
        self.scaler  = GradScaler(enabled=self.use_amp)
        self.epoch   = 0
        self.step    = 0

    # ── Setup ─────────────────────────────────────────────────────────────

    def _setup_dirs(self):
        tr = self.cfg['training']
        self.log_dir  = Path(tr.get('log_dir', 'runs')) / 'mtl_autonomous'
        self.save_dir = Path(tr.get('save_dir', 'checkpoints'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _build_model(self):
        mcfg = self.cfg['model']
        dcfg = self.cfg['dataset']
        tcfg = self.cfg['tasks']

        model_cfg = MTLConfig(
            bev_channels     = mcfg['bev_channels'],
            bev_size         = tuple(dcfg['bev_size']),
            bev_range        = tuple(dcfg['bev_range']),
            voxel_size       = tuple(dcfg['voxel_size']),
            camera_backbone  = mcfg['camera_backbone'],
            img_pretrained   = mcfg.get('img_pretrained', True),
            fusion_layers    = mcfg.get('fusion_layers', 4),
            det_num_classes  = tcfg['detection']['num_classes'],
            seg_num_classes  = tcfg['segmentation']['num_classes'],
            use_detection    = tcfg['detection']['enabled'],
            use_segmentation = tcfg['segmentation']['enabled'],
            use_depth        = tcfg['depth']['enabled'],
        )

        self.model = MTLAutonomousModel(model_cfg).to(self.device)

        # Multi-GPU
        if torch.cuda.device_count() > 1:
            print(f"[Trainer] Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

    def _build_datasets(self):
        dcfg = self.cfg['dataset']
        tr   = self.cfg['training']

        ds_kwargs = dict(
            dataroot  = dcfg['root'],
            version   = dcfg['version'],
            img_size  = tuple(dcfg['img_size']),
            bev_size  = tuple(dcfg['bev_size']),
            bev_range = tuple(dcfg['bev_range']),
        )

        self.train_ds = NuScenesMTL(split='train', **ds_kwargs)
        self.val_ds   = NuScenesMTL(split='val',   **ds_kwargs)

        loader_kwargs = dict(
            batch_size  = tr['batch_size'],
            num_workers = self.cfg['system']['num_workers'],
            collate_fn  = collate_fn,
            pin_memory  = self.device.type == 'cuda',
        )

        self.train_loader = DataLoader(self.train_ds, shuffle=True,  **loader_kwargs)
        self.val_loader   = DataLoader(self.val_ds,   shuffle=False, **loader_kwargs)

        print(f"[Trainer] Train: {len(self.train_ds)} | Val: {len(self.val_ds)}")

    def _build_optimizer(self):
        tr  = self.cfg['training']
        lr  = float(tr['lr'])
        wd  = float(tr['weight_decay'])

        m = self.model.module if hasattr(self.model, 'module') else self.model

        # Freeze everything except seg head
        for param in m.parameters():
            param.requires_grad = False
        for param in m.seg_head.parameters():
            param.requires_grad = True

        print("[Trainer] Frozen all layers — only training seg_head")

        self.optimizer = torch.optim.AdamW(
            m.seg_head.parameters(), lr=lr, weight_decay=wd
        )

        # Flat LR — no warmup needed for single head
        total_steps  = len(self.train_loader) * tr['epochs']
        warmup_steps = 0

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _build_loss(self):
        tr    = self.cfg['training']
        tcfg  = self.cfg['tasks']
        tasks = [t for t in ['detection', 'segmentation', 'depth']
                 if tcfg[t]['enabled']]

        self.loss_fn = MTLLoss(
            task_names = tasks,
            balancer   = tr.get('loss_balancer', 'uncertainty'),
        ).to(self.device)

    # ── Training loop ─────────────────────────────────────────────────────

    def train(self):
        epochs    = self.cfg['training']['epochs']
        log_every = self.cfg['training'].get('log_every', 50)
        val_every = self.cfg['training'].get('val_every', 1)

        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            self._train_epoch(log_every)

            if (epoch + 1) % val_every == 0:
                try:
                    val_loss = self._val_epoch()
                except Exception as e:
                    print(f"  Val skipped: {e}")
                    val_loss = 0.0
                self.writer.add_scalar('val/total_loss', val_loss, epoch)
                self._save_checkpoint(f'epoch_{epoch+1:03d}.pth')

        self.writer.close()
        print("[Trainer] Training complete.")

    def _train_epoch(self, log_every: int):
        self.model.train()
        self.loss_fn.train()

        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}", leave=False)
        for batch in pbar:
            loss, info = self._forward_step(batch, train=True)

            epoch_loss += loss.item()
            self.step  += 1

            if self.step % log_every == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/total_loss', loss.item(), self.step)
                self.writer.add_scalar('train/lr', lr, self.step)
                for k, v in info.items():
                    self.writer.add_scalar(f'train/{k}', v, self.step)

                pbar.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'lr':   f'{lr:.2e}',
                })

        elapsed = time.time() - t0
        print(f"  Epoch {self.epoch+1} | Loss: {epoch_loss/len(self.train_loader):.4f} | {elapsed:.0f}s")

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        self.loss_fn.eval()
        total = 0.0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            loss, _ = self._forward_step(batch, train=False)
            total  += loss.item()

        avg = total / max(len(self.val_loader), 1)
        print(f"  Val loss: {avg:.4f}")
        return avg

    def _forward_step(self, batch: dict, train: bool) -> tuple:
        images     = batch['images'].to(self.device)
        intrinsics = batch['intrinsics'].to(self.device)
        extrinsics = batch['extrinsics'].to(self.device)
        points     = batch['points']                    # list of CPU tensors

        # Move detection targets to device
        targets = {}
        det_t = batch['targets']['detection']
        targets['detection'] = {k: v.to(self.device) for k, v in det_t.items()}
        targets['segmentation'] = batch['targets']['segmentation'].to(self.device)
        targets['depth']        = batch['targets']['depth'].to(self.device).reshape(
            -1, 1, batch['targets']['depth'].shape[-2], batch['targets']['depth'].shape[-1]
        )

        if train:
            self.optimizer.zero_grad()

        with autocast('cuda',enabled=self.use_amp):
            preds = self.model(images, intrinsics, extrinsics, points)
            loss, info = self.loss_fn(preds, targets)

        if train:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg['training'].get('grad_clip', 35.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

        return loss, info

    # ── Checkpointing ─────────────────────────────────────────────────────

    def _save_checkpoint(self, name: str):
        m = self.model.module if hasattr(self.model, 'module') else self.model
        ckpt = {
            'epoch':      self.epoch,
            'step':       self.step,
            'model':      m.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'scheduler':  self.scheduler.state_dict(),
            'loss_fn':    self.loss_fn.state_dict(),
        }
        path = self.save_dir / name
        torch.save(ckpt, path)
        print(f"  Saved checkpoint → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        m    = self.model.module if hasattr(self.model, 'module') else self.model
        m.load_state_dict(ckpt['model'])
        # Skip optimizer/scheduler — new setup for seg-head-only training
        self.epoch = ckpt['epoch'] + 1
        self.step  = ckpt['step']
        print(f"[Trainer] Resumed from {path} (epoch {self.epoch})")
