"""
Unified Multi-Task Learning model for autonomous driving.
Combines camera encoder, LiDAR encoder, cross-modal fusion,
and all task heads into a single forward pass.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from .camera_encoder import CameraEncoder
from .lidar_encoder  import LiDAREncoder
from .fusion         import FusionModule
from .task_heads     import DetectionHead, SegmentationHead, DepthHead


@dataclass
class MTLConfig:
    # BEV parameters
    bev_channels:   int   = 256
    bev_size:       tuple = (200, 200)
    bev_range:      tuple = (-50, -50, 50, 50)    # x_min, y_min, x_max, y_max
    voxel_size:     tuple = (0.5, 0.5, 8.0)
    pc_range:       tuple = (-50, -50, -3, 50, 50, 5)

    # Camera encoder
    camera_backbone: str  = 'swin_tiny'
    img_pretrained:  bool = True

    # Fusion
    fusion_layers:          int   = 4
    fusion_heads:           int   = 8
    fusion_dropout:         float = 0.1

    # Tasks
    det_num_classes: int = 10
    seg_num_classes: int = 14

    # Which tasks to run
    use_detection:    bool = True
    use_segmentation: bool = True
    use_depth:        bool = True


class MTLAutonomousModel(nn.Module):
    """
    Unified LiDAR + Camera multi-task model.

    Forward signature:
        images      (B, N, 3, H, W)
        intrinsics  (B, N, 3, 3)
        extrinsics  (B, N, 4, 4)
        points      list of B tensors, each (N_pts, 4)

    Returns dict with keys: detection, segmentation, depth
    """

    def __init__(self, cfg: MTLConfig = MTLConfig()):
        super().__init__()
        self.cfg = cfg

        # ── Encoders ──────────────────────────────────────────────
        self.camera_encoder = CameraEncoder(
            backbone=cfg.camera_backbone,
            bev_channels=cfg.bev_channels,
            bev_size=cfg.bev_size,
            bev_range=cfg.bev_range,
            pretrained=cfg.img_pretrained,
        )

        self.lidar_encoder = LiDAREncoder(
            bev_channels=cfg.bev_channels,
            bev_size=cfg.bev_size,
            voxel_size=cfg.voxel_size,
            point_cloud_range=cfg.pc_range,
        )

        # ── Cross-modal fusion ────────────────────────────────────
        self.fusion = FusionModule(
            channels=cfg.bev_channels,
            num_cross_attn_layers=cfg.fusion_layers // 2,
            num_self_attn_layers=cfg.fusion_layers // 2,
            num_heads=cfg.fusion_heads,
            dropout=cfg.fusion_dropout,
        )

        # ── Task heads ───────────────────────────────────────────
        if cfg.use_detection:
            self.det_head = DetectionHead(
                in_channels=cfg.bev_channels,
                num_classes=cfg.det_num_classes,
            )

        if cfg.use_segmentation:
            self.seg_head = SegmentationHead(
                in_channels=cfg.bev_channels,
                num_classes=cfg.seg_num_classes,
            )

        if cfg.use_depth:
            self.depth_head = DepthHead(in_channels=cfg.bev_channels)

        self._log_params()

    def _log_params(self):
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MTLModel] Total params:     {total/1e6:.1f}M")
        print(f"[MTLModel] Trainable params: {trainable/1e6:.1f}M")

    def forward(
        self,
        images:     torch.Tensor,         # (B, N, 3, H, W)
        intrinsics: torch.Tensor,         # (B, N, 3, 3)
        extrinsics: torch.Tensor,         # (B, N, 4, 4)
        points:     list[torch.Tensor],   # list of B tensors (N_pts, 4)
    ) -> dict[str, any]:

        outputs = {}

        # ── Camera branch → BEV ───────────────────────────────────
        cam_bev = self.camera_encoder(images, intrinsics, extrinsics)   # (B, C, H, W)

        # ── LiDAR branch → BEV ───────────────────────────────────
        lid_bev = self.lidar_encoder(points)                             # (B, C, H, W)

        # ── Cross-modal fusion ────────────────────────────────────
        fused_bev = self.fusion(cam_bev, lid_bev)                       # (B, C, H, W)

        # ── Task heads ────────────────────────────────────────────
        if self.cfg.use_detection:
            outputs['detection'] = self.det_head(fused_bev)

        if self.cfg.use_segmentation:
            outputs['segmentation'] = self.seg_head(fused_bev)

        if self.cfg.use_depth:
            # Depth runs on per-camera features (not BEV)
            # Re-use the cam BEV as proxy (replace with FPN feats for best quality)
            B, N = images.shape[:2]
            cam_feat_proxy = cam_bev.unsqueeze(1).expand(-1, N, -1, -1, -1)
            cam_feat_proxy = cam_feat_proxy.reshape(B * N, *cam_bev.shape[1:])
            outputs['depth'] = self.depth_head(cam_feat_proxy)   # (B*N, 1, H, W)

        # Also expose intermediate BEV for debugging
        outputs['cam_bev']   = cam_bev
        outputs['lid_bev']   = lid_bev
        outputs['fused_bev'] = fused_bev

        return outputs

    def get_task_parameters(self) -> dict[str, list]:
        """Group parameters by component (useful for per-group LR scheduling)."""
        return {
            'backbone':   list(self.camera_encoder.backbone.parameters()) +
                          list(self.lidar_encoder.parameters()),
            'fusion':     list(self.fusion.parameters()),
            'task_heads': (list(self.det_head.parameters()   if self.cfg.use_detection    else []) +
                           list(self.seg_head.parameters()   if self.cfg.use_segmentation else []) +
                           list(self.depth_head.parameters() if self.cfg.use_depth        else [])),
        }

    @classmethod
    def from_config(cls, cfg_dict: dict) -> 'MTLAutonomousModel':
        cfg = MTLConfig(**{k: v for k, v in cfg_dict.items() if hasattr(MTLConfig, k)})
        return cls(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = MTLConfig(
        camera_backbone='swin_tiny',
        bev_channels=128,       # smaller for quick test
        bev_size=(100, 100),
        bev_range=(-50, -50, 50, 50),
        img_pretrained=False,
    )

    model = MTLAutonomousModel(cfg).to(device)

    B, N = 2, 6
    imgs        = torch.randn(B, N, 3, 224, 400, device=device)
    intrinsics  = torch.eye(3, device=device)[None, None].expand(B, N, -1, -1)
    extrinsics  = torch.eye(4, device=device)[None, None].expand(B, N, -1, -1)
    points      = [torch.randn(5000, 4) for _ in range(B)]

    with torch.no_grad():
        out = model(imgs, intrinsics, extrinsics, points)

    print("\n── Outputs ──")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv.shape}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
