"""
LiDAR encoder: PointPillars architecture
Point cloud (N, 4) → voxelization → PillarFeatureNet → BEV pseudo-image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PillarFeatureNet(nn.Module):
    """
    Per-pillar PointNet: encodes all points in a voxel pillar into a fixed vector.
    Input:  (B, num_pillars, max_pts_per_pillar, D_in)
    Output: (B, num_pillars, C_out)
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 64, num_filters: int = 64):
        super().__init__()
        # Features per point: x, y, z, intensity, Δx, Δy, Δz from pillar centre, x_c, y_c
        self.net = nn.Sequential(
            nn.Linear(in_channels, num_filters, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True),
            nn.Linear(num_filters, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, pillars: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """
        pillars:    (B, P, K, D)  — B batches, P pillars, K points/pillar, D features
        num_points: (B, P)        — actual point count per pillar (for masking)
        """
        B, P, K, D = pillars.shape

        # Flatten for BatchNorm1d
        flat = pillars.reshape(B * P * K, D)
        encoded = self.net(flat)                   # (B*P*K, C)
        encoded = encoded.reshape(B, P, K, -1)

        # Mask padded points
        mask = torch.arange(K, device=pillars.device).unsqueeze(0) < num_points.unsqueeze(-1)  # (B*P, K)
        mask = mask.reshape(B, P, K).unsqueeze(-1).float()
        encoded = encoded * mask

        # Max-pool over K points → pillar descriptor (B, P, C)
        pillar_feat = encoded.max(dim=2).values
        return pillar_feat


class PointPillarsScatter(nn.Module):
    """
    Scatter pillar features back into a 2D pseudo-image (BEV).
    Input:  pillar features (B, P, C) + pillar indices (B, P, 2)
    Output: BEV image (B, C, H, W)
    """

    def __init__(self, bev_h: int, bev_w: int, channels: int):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.channels = channels

    def forward(
        self,
        pillar_feats: torch.Tensor,   # (B, P, C)
        pillar_xy: torch.Tensor,      # (B, P, 2)  integer BEV coords (col, row)
        pillar_mask: torch.Tensor,    # (B, P)     valid pillar flag
    ) -> torch.Tensor:                # (B, C, H, W)

        B, P, C = pillar_feats.shape
        bev = torch.zeros(B, C, self.bev_h, self.bev_w,
                          device=pillar_feats.device, dtype=pillar_feats.dtype)

        for b in range(B):
            valid = pillar_mask[b]                     # (P,)
            xy = pillar_xy[b][valid]                   # (P', 2)
            feats = pillar_feats[b][valid]             # (P', C)

            col = xy[:, 0].clamp(0, self.bev_w - 1)
            row = xy[:, 1].clamp(0, self.bev_h - 1)

            # Scatter: last-write wins (sufficient for non-overlapping pillars)
            bev[b, :, row, col] = feats.T

        return bev


class BackboneNeck(nn.Module):
    """
    2D CNN backbone + FPN neck on top of the BEV pseudo-image.
    Produces a single unified BEV feature map at the original resolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Strided encoder
        self.block1 = self._make_block(in_channels, 64, stride=2)
        self.block2 = self._make_block(64, 128, stride=2)
        self.block3 = self._make_block(128, 256, stride=2)

        # FPN decoders (upsample back to original BEV res)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(64, out_channels // 2, 2, stride=2),
                                  nn.BatchNorm2d(out_channels // 2), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, out_channels // 2, 4, stride=4),
                                  nn.BatchNorm2d(out_channels // 2), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, out_channels // 2, 8, stride=8),
                                  nn.BatchNorm2d(out_channels // 2), nn.ReLU(inplace=True))

        self.merge = nn.Sequential(
            nn.Conv2d(out_channels // 2 * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _make_block(in_c: int, out_c: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)

        u1 = self.up1(c1)
        u2 = self.up2(c2)
        u3 = self.up3(c3)

        # Align spatial sizes (u2, u3 might differ by 1 px at some resolutions)
        h, w = u1.shape[2], u1.shape[3]
        u2 = F.interpolate(u2, size=(h, w), mode='bilinear', align_corners=False)
        u3 = F.interpolate(u3, size=(h, w), mode='bilinear', align_corners=False)

        return self.merge(torch.cat([u1, u2, u3], dim=1))


class Voxelizer(nn.Module):
    """
    CPU-based voxelizer: converts raw point cloud to pillars.
    In production use mmdet3d or OpenPCDet's CUDA voxelizer.
    """

    def __init__(
        self,
        voxel_size: tuple = (0.5, 0.5, 8.0),
        point_cloud_range: tuple = (-50, -50, -3, 50, 50, 5),
        max_num_points: int = 32,
        max_voxels: int = 16000,
    ):
        super().__init__()
        self.voxel_size = torch.tensor(voxel_size)
        self.pc_range   = torch.tensor(point_cloud_range)
        self.max_pts    = max_num_points
        self.max_voxels = max_voxels

        grid_size = (self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size
        self.grid_size = grid_size.long()  # (X, Y, Z)

    @torch.no_grad()
    def forward(self, points: torch.Tensor) -> tuple:
        """
        points: (N, 4) — x, y, z, intensity (single-sample, on CPU)
        Returns:
            voxels      (P, K, 9)   pillar features
            coords      (P, 3)      voxel grid indices (z, y, x)
            num_points  (P,)        valid points per pillar
        """
        device = points.device

        # Filter to range
        pc_range = self.pc_range.to(device)
        vs       = self.voxel_size.to(device)

        mask = ((points[:, :3] >= pc_range[:3]) &
                (points[:, :3] <  pc_range[3:])).all(dim=1)
        points = points[mask]

        if points.shape[0] == 0:
            empty = torch.zeros(1, self.max_pts, 9, device=device)
            return empty, torch.zeros(1, 3, dtype=torch.long, device=device), torch.ones(1, dtype=torch.long, device=device)

        # Voxel indices
        coords_f = (points[:, :3] - pc_range[:3]) / vs     # (N, 3) float
        coords_i = coords_f.long()                          # (N, 3) int
        coords_i = coords_i.clamp_max(self.grid_size.to(device) - 1)

        # Unique voxels
        flat = coords_i[:, 0] * self.grid_size[1].to(device) * self.grid_size[2].to(device) + \
               coords_i[:, 1] * self.grid_size[2].to(device) + coords_i[:, 2]

        uniq, inverse = torch.unique(flat, return_inverse=True)
        num_voxels = min(len(uniq), self.max_voxels)
        uniq = uniq[:num_voxels]

        voxels     = torch.zeros(num_voxels, self.max_pts, 9, device=device)
        num_points = torch.zeros(num_voxels, dtype=torch.long, device=device)
        vox_coords = torch.zeros(num_voxels, 3, dtype=torch.long, device=device)

        for vi in range(num_voxels):
            pt_mask  = (inverse == vi)
            pts_in   = points[pt_mask]
            k        = min(pts_in.shape[0], self.max_pts)
            pts_in   = pts_in[:k]
            cx, cy   = pts_in[:, 0].mean(), pts_in[:, 1].mean()
            cz       = pts_in[:, 2].mean()

            feat = torch.zeros(k, 9, device=device)
            feat[:, :4] = pts_in[:, :4]           # x, y, z, intensity
            feat[:, 4]  = pts_in[:, 0] - cx       # Δx from pillar centre
            feat[:, 5]  = pts_in[:, 1] - cy       # Δy
            feat[:, 6]  = pts_in[:, 2] - cz       # Δz
            feat[:, 7]  = cx
            feat[:, 8]  = cy

            voxels[vi, :k] = feat
            num_points[vi] = k
            vox_coords[vi] = coords_i[pt_mask][0]

        return voxels, vox_coords, num_points


class LiDAREncoder(nn.Module):
    """
    Full LiDAR encoder: raw point cloud → BEV feature map
    PointPillars pipeline (Voxelizer → PillarFeatureNet → Scatter → BackboneNeck)
    """

    def __init__(
        self,
        bev_channels: int = 256,
        bev_size: tuple = (200, 200),
        voxel_size: tuple = (0.5, 0.5, 8.0),
        point_cloud_range: tuple = (-50, -50, -3, 50, 50, 5),
        max_num_points: int = 32,
        max_voxels: int = 16000,
    ):
        super().__init__()
        self.bev_h, self.bev_w = bev_size

        self.voxelizer = Voxelizer(voxel_size, point_cloud_range, max_num_points, max_voxels)

        pillar_channels = 64
        self.pillar_net = PillarFeatureNet(
            in_channels=9,
            out_channels=pillar_channels,
        )
        self.scatter = PointPillarsScatter(self.bev_h, self.bev_w, pillar_channels)
        self.backbone_neck = BackboneNeck(pillar_channels, bev_channels)

        # BEV coords from voxel indices
        pc_range = torch.tensor(point_cloud_range)
        vs       = torch.tensor(voxel_size)
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('voxel_size', vs)

    def _coords_to_bev_xy(self, coords: torch.Tensor) -> torch.Tensor:
        """Convert voxel grid indices to BEV pixel coords (col, row)."""
        # coords: (P, 3) → (x_idx, y_idx, z_idx)
        bev_col = coords[:, 0]  # x direction
        bev_row = coords[:, 1]  # y direction
        return torch.stack([bev_col, bev_row], dim=1)

    def forward(self, points_list: list[torch.Tensor]) -> torch.Tensor:
        """
        points_list: list of B tensors, each (N_i, 4) on CPU
        Returns: (B, bev_channels, bev_H, bev_W)
        """
        B = len(points_list)
        device = next(self.parameters()).device

        all_pillar_feats = []
        all_pillar_xy    = []
        all_pillar_mask  = []

        for pts in points_list:
            voxels, coords, num_pts = self.voxelizer(pts.cpu())
            voxels  = voxels.to(device)
            coords  = coords.to(device)
            num_pts = num_pts.to(device)

            P = voxels.shape[0]
            # PillarFeatureNet expects (1, P, K, D)
            pf = self.pillar_net(voxels.unsqueeze(0), num_pts.unsqueeze(0))  # (1, P, 64)
            pf = pf.squeeze(0)   # (P, 64)

            bev_xy = self._coords_to_bev_xy(coords)     # (P, 2)
            mask   = torch.ones(P, dtype=torch.bool, device=device)

            all_pillar_feats.append(pf)
            all_pillar_xy.append(bev_xy)
            all_pillar_mask.append(mask)

        # Pad to same P and batch
        max_P = max(f.shape[0] for f in all_pillar_feats)
        C     = all_pillar_feats[0].shape[1]

        batch_feats = torch.zeros(B, max_P, C, device=device)
        batch_xy    = torch.zeros(B, max_P, 2, dtype=torch.long, device=device)
        batch_mask  = torch.zeros(B, max_P, dtype=torch.bool, device=device)

        for i, (pf, xy, m) in enumerate(zip(all_pillar_feats, all_pillar_xy, all_pillar_mask)):
            p = pf.shape[0]
            batch_feats[i, :p] = pf
            batch_xy[i, :p]    = xy
            batch_mask[i, :p]  = m

        # Scatter → BEV pseudo-image
        bev_img = self.scatter(batch_feats, batch_xy, batch_mask)   # (B, 64, H, W)

        # 2D backbone + neck → rich BEV features
        bev_out = self.backbone_neck(bev_img)                         # (B, 256, H, W)

        return bev_out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = LiDAREncoder(bev_channels=256, bev_size=(200, 200)).to(device)

    # Fake batch of 2 point clouds
    pts_list = [
        torch.randn(12000, 4),   # sample 0: 12k points
        torch.randn(8000,  4),   # sample 1: 8k points
    ]
    bev = encoder(pts_list)
    print(f"LiDAR BEV output: {bev.shape}")  # (2, 256, 200, 200)
