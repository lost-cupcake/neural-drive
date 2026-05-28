"""Windowed attention fusion — matches Kaggle T4 trained checkpoint."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    def __init__(self, channels, num_heads=8, window_size=10, dropout=0.1):
        super().__init__()
        self.ws = window_size
        self.attn = nn.MultiheadAttention(channels, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels), nn.Dropout(dropout),
        )

    def _to_windows(self, x):
        B, C, H, W = x.shape
        ws = self.ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, Hp//ws, ws, Wp//ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        nw = (Hp//ws) * (Wp//ws)
        return x.reshape(B * nw, ws*ws, C), B, Hp, Wp

    def _from_windows(self, x, B, Hp, Wp, H, W):
        ws = self.ws
        C  = x.shape[-1]
        x  = x.reshape(B, Hp//ws, Wp//ws, ws, ws, C)
        x  = x.permute(0, 1, 3, 2, 4, 5)
        x  = x.reshape(B, Hp, Wp, C)
        return x[:, :H, :W, :].permute(0, 3, 1, 2)

    def forward(self, x, kv=None):
        B, C, H, W = x.shape
        q_wins, B, Hp, Wp = self._to_windows(x)
        kv_wins = self._to_windows(kv)[0] if kv is not None else q_wins
        qn  = self.norm1(q_wins)
        kvn = self.norm1(kv_wins)
        out, _ = self.attn(qn, kvn, kvn)
        q_wins = q_wins + out
        q_wins = q_wins + self.ffn(self.norm2(q_wins))
        return self._from_windows(q_wins, B, Hp, Wp, H, W)


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, cam, lid):
        w = self.gate(torch.cat([cam, lid], dim=1))
        return w[:, 0:1] * cam + w[:, 1:2] * lid


class FusionModule(nn.Module):
    def __init__(self, channels=256, num_cross_attn_layers=2,
                 num_self_attn_layers=2, num_heads=8,
                 dropout=0.1, window_size=10):
        super().__init__()
        self.adaptive_gate = AdaptiveFeatureFusion(channels)
        self.cross_attn_layers = nn.ModuleList([
            WindowAttention(channels, num_heads, window_size, dropout)
            for _ in range(num_cross_attn_layers)
        ])
        self.self_attn_layers = nn.ModuleList([
            WindowAttention(channels, num_heads, window_size, dropout)
            for _ in range(num_self_attn_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cam_bev, lid_bev):
        if cam_bev.shape != lid_bev.shape:
            lid_bev = F.interpolate(lid_bev, size=cam_bev.shape[2:],
                                    mode='bilinear', align_corners=False)
        fused = self.adaptive_gate(cam_bev, lid_bev)
        for layer in self.cross_attn_layers:
            fused = layer(fused, lid_bev)
        for layer in self.self_attn_layers:
            fused = layer(fused)
        return self.output_proj(fused)


class MultiScaleFusion(nn.Module):
    def __init__(self, channels=256, scales=3):
        super().__init__()
        self.scales = scales
        self.downsamplers = nn.ModuleList([
            nn.AvgPool2d(2**i, stride=2**i) for i in range(1, scales)
        ])
        self.fusions = nn.ModuleList([
            FusionModule(channels, 1, 1) for _ in range(scales)
        ])
        self.merge = nn.Sequential(
            nn.Conv2d(channels * scales, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cam_bev, lid_bev):
        H, W = cam_bev.shape[2], cam_bev.shape[3]
        feats = [self.fusions[0](cam_bev, lid_bev)]
        for down, fusion in zip(self.downsamplers, self.fusions[1:]):
            f = fusion(down(cam_bev), down(lid_bev))
            feats.append(F.interpolate(f, (H, W), mode='bilinear',
                                       align_corners=False))
        return self.merge(torch.cat(feats, dim=1))