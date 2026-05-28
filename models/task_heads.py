"""
Task heads — each operates on the shared BEV feature map.

  DetectionHead   → 3D bounding boxes (CenterPoint-style heatmap)
  SegmentationHead → BEV semantic segmentation (drivable, lane, etc.)
  DepthHead        → per-pixel depth from camera perspective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def conv_bn_relu(in_c: int, out_c: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3D Detection Head  (CenterPoint-style)
# ──────────────────────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    CenterPoint-style detection head on BEV features.

    Outputs per-BEV-cell predictions:
      • heatmap    (num_classes,)   — Gaussian peak at object centre
      • offset     (2,)             — sub-voxel centre offset (x, y)
      • wlh        (3,)             — width, length, height in metres
      • yaw        (2,)             — sin/cos of heading angle
      • velocity   (2,)             — vx, vy in m/s

    During inference, peaks in the heatmap are decoded to 3D boxes.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,      # nuScenes: car, truck, bus, ped, cyclist, …
        hidden_channels: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Shared feature extractor
        self.shared = nn.Sequential(
            conv_bn_relu(in_channels, hidden_channels),
            conv_bn_relu(hidden_channels, hidden_channels),
        )

        # Heatmap (classification) — sigmoid output
        self.heatmap = nn.Sequential(
            conv_bn_relu(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )
        # Sub-voxel centre offset
        self.offset = nn.Sequential(
            conv_bn_relu(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 2, 1),
        )
        # Box dimensions (w, l, h)
        self.wlh = nn.Sequential(
            conv_bn_relu(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 3, 1),
        )
        # Yaw as (sin, cos) — avoids angle discontinuities
        self.yaw = nn.Sequential(
            conv_bn_relu(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 2, 1),
        )
        # Velocity
        self.velocity = nn.Sequential(
            conv_bn_relu(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Heatmap bias init (focal-loss prior: very few foreground cells)
        nn.init.constant_(self.heatmap[-1].bias, -2.19)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.heatmap[-1]:
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, bev: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        bev: (B, C, H, W)
        Returns dict with keys: heatmap, offset, wlh, yaw, velocity
        """
        feat = self.shared(bev)

        return {
            'heatmap':  self.heatmap(feat).sigmoid(),    # (B, cls, H, W) [0,1]
            'offset':   self.offset(feat),               # (B, 2,   H, W)
            'wlh':      self.wlh(feat).exp(),            # (B, 3,   H, W) positive
            'yaw':      self.yaw(feat),                  # (B, 2,   H, W) sin/cos
            'velocity': self.velocity(feat),             # (B, 2,   H, W)
        }

    @torch.no_grad()
    def decode_boxes(
        self,
        preds: dict,
        bev_range: tuple,
        score_thresh: float = 0.3,
        nms_thresh: float   = 0.5,
        topk: int           = 500,
    ) -> list[dict]:
        """
        Convert raw head outputs to a list of 3D boxes per batch item.
        bev_range: (x_min, y_min, x_max, y_max)
        """
        B, cls, H, W = preds['heatmap'].shape
        x_min, y_min, x_max, y_max = bev_range
        dx = (x_max - x_min) / W
        dy = (y_max - y_min) / H

        results = []
        for b in range(B):
            hm   = preds['heatmap'][b]     # (cls, H, W)
            off  = preds['offset'][b]      # (2, H, W)
            wlh  = preds['wlh'][b]         # (3, H, W)
            yaw  = preds['yaw'][b]         # (2, H, W)
            vel  = preds['velocity'][b]    # (2, H, W)

            scores, class_ids = hm.max(dim=0)   # (H, W)
            scores_flat = scores.flatten()
            topk_scores, topk_idx = scores_flat.topk(min(topk, scores_flat.numel()))

            keep = topk_scores > score_thresh
            topk_idx    = topk_idx[keep]
            topk_scores = topk_scores[keep]

            rows = topk_idx // W
            cols = topk_idx  % W

            cx = x_min + (cols.float() + 0.5 + off[0].flatten()[topk_idx]) * dx
            cy = y_min + (rows.float() + 0.5 + off[1].flatten()[topk_idx]) * dy

            w_ = wlh[0].flatten()[topk_idx]
            l_ = wlh[1].flatten()[topk_idx]
            h_ = wlh[2].flatten()[topk_idx]

            sin_yaw = yaw[0].flatten()[topk_idx]
            cos_yaw = yaw[1].flatten()[topk_idx]
            angle   = torch.atan2(sin_yaw, cos_yaw)

            results.append({
                'boxes_3d': torch.stack([cx, cy, w_, l_, h_, angle], dim=1),  # (N, 6)
                'scores':   topk_scores,
                'labels':   class_ids.flatten()[topk_idx],
                'velocity': torch.stack([vel[0].flatten()[topk_idx],
                                         vel[1].flatten()[topk_idx]], dim=1),
            })
        return results


# ──────────────────────────────────────────────────────────────────────────────
# BEV Segmentation Head
# ──────────────────────────────────────────────────────────────────────────────

class SegmentationHead(nn.Module):
    """
    BEV semantic segmentation — simple encoder-decoder (no skip connections).
    Avoids spatial-size mismatch issues; sufficient for BEV seg at training scale.
    """

    NUSCENES_MAP_CLASSES = [
        'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
        'carpark_area', 'divider', 'lane_divider',
        'road_block', 'bike_lane', 'traffic_cone_zone',
        'construction_zone', 'vegetation', 'terrain', 'background',
    ]

    def __init__(self, in_channels: int = 256, num_classes: int = 14):
        super().__init__()
        self.num_classes = num_classes

        self.decoder = nn.Sequential(
            conv_bn_relu(in_channels, 128),
            conv_bn_relu(128, 128),
            conv_bn_relu(128, 64),
            conv_bn_relu(64, 64),
        )
        self.classifier = nn.Conv2d(64, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        bev: (B, C, H, W)
        Returns: (B, num_classes, H, W) — raw logits
        """
        x = self.decoder(bev)
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────────────────────
# Depth Estimation Head
# ──────────────────────────────────────────────────────────────────────────────

class DepthHead(nn.Module):
    """
    Monocular depth estimation in the camera perspective (not BEV).

    Takes multi-scale camera features (from FPN) instead of the BEV,
    so it runs in parallel on the camera encoder output.

    Outputs: (B*N, 1, H_img, W_img) — per-pixel depth in metres
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            conv_bn_relu(in_channels, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            conv_bn_relu(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            conv_bn_relu(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            conv_bn_relu(32, 16),
        )
        self.depth_out = nn.Sequential(
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Softplus(),   # positive depth
        )

    def forward(self, cam_feats: torch.Tensor) -> torch.Tensor:
        """
        cam_feats: (B*N, C, H_feat, W_feat)
        Returns:   (B*N, 1, H_out, W_out)
        """
        x = self.decoder(cam_feats)
        return self.depth_out(x)


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bev = torch.randn(2, 256, 200, 200, device=device)

    det = DetectionHead(num_classes=10).to(device)
    seg = SegmentationHead(num_classes=14).to(device)

    det_out = det(bev)
    seg_out = seg(bev)

    print("Detection outputs:")
    for k, v in det_out.items():
        print(f"  {k}: {v.shape}")

    print(f"Segmentation output: {seg_out.shape}")   # (2, 14, 200, 200)

    # Depth head (camera perspective)
    cam_feats = torch.randn(12, 256, 112, 200, device=device)  # B*N=2*6=12
    depth = DepthHead().to(device)
    depth_out = depth(cam_feats)
    print(f"Depth output: {depth_out.shape}")         # (12, 1, ~H_img, ~W_img)
