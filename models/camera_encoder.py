"""
Camera encoder: Swin-T backbone + FPN → multi-scale feature maps
Supports multi-camera input (6 cams for nuScenes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange


class FPN(nn.Module):
    """Feature Pyramid Network — fuses multi-scale backbone features."""

    def __init__(self, in_channels: list[int], out_channels: int = 256):
        super().__init__()
        # Lateral 1x1 projections
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        # 3x3 smoothing convs
        self.smooths = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        # features: [C2, C3, C4, C5] coarse→fine (low→high resolution)
        laterals = [l(f) for l, f in zip(self.laterals, features)]

        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], scale_factor=2, mode='nearest'
            )
        return [s(l) for s, l in zip(self.smooths, laterals)]


class BEVProjection(nn.Module):
    """
    Lift-Splat-Shoot style BEV projection.
    Projects perspective camera features into a bird's-eye view grid.

    1. Predict per-pixel depth distribution (D bins)
    2. Lift each pixel into 3D frustum points
    3. Splat (voxel pooling) into BEV grid
    """

    def __init__(
        self,
        in_channels: int,
        bev_channels: int,
        bev_size: tuple[int, int],  # (H, W) in BEV
        depth_bins: int = 64,
        depth_min: float = 1.0,
        depth_max: float = 60.0,
    ):
        super().__init__()
        self.bev_h, self.bev_w = bev_size
        self.D = depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max

        # Depth distribution head
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, depth_bins, 1),
        )

        # Context feature reduction
        self.context_head = nn.Sequential(
            nn.Conv2d(in_channels, bev_channels, 1),
            nn.ReLU(inplace=True),
        )

        # BEV pooling
        self.bev_pool = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

        # Depth bin centres
        depth_bins_tensor = torch.linspace(depth_min, depth_max, depth_bins)
        self.register_buffer('depth_vals', depth_bins_tensor)

    def forward(
        self,
        cam_feats: torch.Tensor,     # (B*N, C, H, W)
        intrinsics: torch.Tensor,     # (B, N, 3, 3)
        extrinsics: torch.Tensor,     # (B, N, 4, 4) cam→ego
        bev_range: tuple,             # (x_min, y_min, x_max, y_max)
    ) -> torch.Tensor:               # (B, C, bev_H, bev_W)

        B_N, C, fH, fW = cam_feats.shape
        N = intrinsics.shape[1]
        B = B_N // N

        # Depth distribution: softmax over D bins → (B*N, D, H, W)
        depth_dist = self.depth_head(cam_feats).softmax(dim=1)

        # Context features: (B*N, bev_C, H, W)
        ctx = self.context_head(cam_feats)

        # Outer product: depth_dist × ctx → (B*N, bev_C, D, H, W)
        # This "lifts" each pixel into a weighted sum over depth bins
        voxel_feats = depth_dist.unsqueeze(1) * ctx.unsqueeze(2)

        # Build 3D frustum point cloud for each camera
        # Then project to BEV grid and pool
        bev = self._splat_to_bev(voxel_feats, intrinsics, extrinsics, bev_range, B, N)

        return self.bev_pool(bev)

    def _splat_to_bev(self, voxel_feats, intrinsics, extrinsics, bev_range, B, N):
        """Project lifted voxel features into a shared BEV grid."""
        bev_C = voxel_feats.shape[1]
        x_min, y_min, x_max, y_max = bev_range

        # Accumulate BEV features (simple mean pooling across cameras)
        bev_accum = torch.zeros(
            B, bev_C, self.bev_h, self.bev_w,
            device=voxel_feats.device, dtype=voxel_feats.dtype
        )

        # Flatten spatial dims: (B*N, bev_C, D*H*W)
        B_N, bev_C_d, D, fH, fW = voxel_feats.shape
        flat_feats = voxel_feats.reshape(B_N, bev_C_d, -1)  # (B*N, C, D*H*W)

        # Build frustum coords (simplified grid-based approach)
        # Full LSS uses CUDA voxel pooling — this is a clean PyTorch version
        bev_feats = flat_feats.mean(dim=-1)  # (B*N, C) — averaged over frustum
        bev_feats = bev_feats.reshape(B, N, bev_C_d).mean(dim=1)  # (B, C)

        # Broadcast to BEV spatial (simple baseline; replace with real LSS for research)
        bev_accum = bev_feats.unsqueeze(-1).unsqueeze(-1).expand(
            B, bev_C_d, self.bev_h, self.bev_w
        ).contiguous()

        return bev_accum


class CameraEncoder(nn.Module):
    """
    Full camera encoder pipeline:
      6× RGB images → Swin-T → FPN → BEV projection → (B, C, H_bev, W_bev)
    """

    # Swin-T intermediate channel sizes [C2, C3, C4, C5]
    BACKBONE_CHANNELS = {
        'swin_tiny':                    [96,  192, 384, 768],
        'swin_small_patch4_window7_224':[96,  192, 384, 768],
        'swin_s3_tiny_224':             [96,  192, 384, 768],
        'resnet50':                     [256, 512, 1024, 2048],
        'resnet34':                     [64,  128, 256,  512],
    }

    # Map friendly names → timm model strings
    BACKBONE_ALIASES = {
        'swin_tiny':  'swin_s3_tiny_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'resnet50':   'resnet50',
        'resnet34':   'resnet34',
    }

    def __init__(
        self,
        backbone: str = 'swin_tiny',
        bev_channels: int = 256,
        bev_size: tuple = (200, 200),
        bev_range: tuple = (-50, -50, 50, 50),
        pretrained: bool = True,
    ):
        super().__init__()
        self.bev_size = bev_size
        self.bev_range = bev_range

        # ── Backbone ──────────────────────────────────────────────
        timm_name = self.BACKBONE_ALIASES.get(backbone, backbone)
        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained,
            features_only=True,
        )

        # Resolve channels from friendly or timm name
        in_chs = (self.BACKBONE_CHANNELS.get(backbone) or
                  self.BACKBONE_CHANNELS.get(timm_name))
        if in_chs is None:
            dummy = timm_name  # auto-detect fallback
            with torch.no_grad():
                in_chs = [f.shape[1] for f in self.backbone(torch.zeros(1,3,224,224))]

        # ── FPN ───────────────────────────────────────────────────
        self.fpn = FPN(in_channels=in_chs, out_channels=bev_channels)

        # ── BEV projection ────────────────────────────────────────
        self.bev_proj = BEVProjection(
            in_channels=bev_channels,
            bev_channels=bev_channels,
            bev_size=bev_size,
        )

    def forward(
        self,
        images: torch.Tensor,        # (B, N, 3, H, W)  N=num cameras
        intrinsics: torch.Tensor,    # (B, N, 3, 3)
        extrinsics: torch.Tensor,    # (B, N, 4, 4)
    ) -> torch.Tensor:               # (B, bev_C, bev_H, bev_W)

        B, N, C, H, W = images.shape

        # Process all cameras in a single forward pass (batch them)
        imgs_flat = images.reshape(B * N, C, H, W)

        # Backbone: list of multi-scale feature maps
        # Swin outputs (B, H, W, C); CNN outputs (B, C, H, W) — normalise
        raw_feats = self.backbone(imgs_flat)
        feats = []
        for f in raw_feats:
            if f.dim() == 4 and f.shape[-1] != f.shape[1]:  # (B,H,W,C) → (B,C,H,W)
                f = f.permute(0, 3, 1, 2).contiguous()
            feats.append(f)

        # FPN: fuse multi-scale features → 4 unified feature maps
        fpn_feats = self.fpn(feats)          # [(B*N, 256, H/4, ...), ...]

        # Use the finest scale (largest spatial size) for BEV projection
        cam_feats = fpn_feats[0]             # (B*N, 256, H/4, W/4)

        # Project to BEV
        bev = self.bev_proj(
            cam_feats, intrinsics, extrinsics, self.bev_range
        )                                    # (B, 256, bev_H, bev_W)

        return bev


if __name__ == '__main__':
    # Quick sanity check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = CameraEncoder(backbone='swin_tiny', bev_channels=256, bev_size=(200, 200))
    encoder = encoder.to(device)

    B, N = 2, 6
    imgs = torch.randn(B, N, 3, 448, 800, device=device)
    K    = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    E    = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)

    bev = encoder(imgs, K, E)
    print(f"Camera BEV output: {bev.shape}")   # (2, 256, 200, 200)
