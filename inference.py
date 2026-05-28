import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

sys.path.insert(0, ".")
from models.mtl_model import MTLAutonomousModel, MTLConfig
from datasets.nuscenes_mtl import NuScenesMTL

CKPT     = "checkpoints/epoch_024_T4_BEST.pth"
DATAROOT = "D:/downloads/v1.0-mini"
BG       = "#060d14"
PBG      = "#0d1117"

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = MTLConfig(
    camera_backbone="swin_tiny",
    bev_channels=128,
    bev_size=(200, 200),
    bev_range=(-50, -50, 50, 50),
    img_pretrained=False,
    use_detection=True,
    use_segmentation=True,
    use_depth=False,
    det_num_classes=10,
    seg_num_classes=14,
    fusion_layers=2,
    fusion_heads=8,
)
model = MTLAutonomousModel(cfg).to(device)
ckpt  = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded {CKPT} | device={device}")

print("Finding best sample...")
ds = NuScenesMTL(dataroot=DATAROOT, version="v1.0-mini", split="train",
                 img_size=(224,224), bev_size=(200,200), bev_range=(-50,-50,50,50))

best_idx, best_n = 0, 0
for i in range(min(30, len(ds))):
    s  = ds[i]
    gt = s["targets"]["segmentation"].numpy()
    hm = s["targets"]["detection"]["heatmap"]
    n  = len(np.unique(gt)) + int(hm.max() > 0) * 3
    if n > best_n:
        best_n = n; best_idx = i

sample = ds[best_idx]
print(f"Sample {best_idx} | {best_n} score | {sample['points'].shape[0]:,} pts")

imgs = sample["images"].unsqueeze(0).to(device)
K    = sample["intrinsics"].unsqueeze(0).to(device)
E    = sample["extrinsics"].unsqueeze(0).to(device)
pts  = [sample["points"]]

print("Inference...")
with torch.no_grad():
    preds = model(imgs, K, E, pts)
print("Done!")

gt_seg  = sample["targets"]["segmentation"].numpy()
heatmap = preds["detection"]["heatmap"][0].max(0).values.cpu().numpy()
H_bev, W_bev = heatmap.shape

hm_min, hm_max = heatmap.min(), heatmap.max()
hm_norm = (heatmap - hm_min) / (hm_max - hm_min + 1e-6)

feat = preds["fused_bev"][0].cpu().float().numpy()
C, H, W = feat.shape
flat    = feat.reshape(C, -1).T
flat   -= flat.mean(0)
flat   /= (flat.std(0) + 1e-6)
U, S, Vt = np.linalg.svd(flat, full_matrices=False)
pca      = U[:, :3]
p2, p98  = np.percentile(pca, 2), np.percentile(pca, 98)
pca      = np.clip((pca - p2) / (p98 - p2 + 1e-6), 0, 1).reshape(H, W, 3)

SEG_COLORS = np.array([
    [34,  197, 94 ], [251, 146, 60 ], [167, 139, 250], [248, 113, 113],
    [56,  189, 248], [20,  184, 166], [129, 140, 248], [18,  18,  18 ],
    [18,18,18],[18,18,18],[18,18,18],[18,18,18],[18,18,18],[18,18,18],
], dtype=np.float32) / 255.0

SEG_NAMES = ["Drivable","Ped Crossing","Walkway","Stop Line",
             "Carpark","Road Div","Lane Div","Background",
             "-","-","-","-","-","-"]

def seg_to_rgb(lmap):
    out = np.zeros((*lmap.shape, 3), dtype=np.float32)
    for c in range(14): out[lmap==c] = SEG_COLORS[c]
    return out

mean_img = np.array([0.485,0.456,0.406])
std_img  = np.array([0.229,0.224,0.225])

thr   = max(float(heatmap.mean()) + float(heatmap.std()), 0.01)
ys,xs = np.where(heatmap > thr)
n_det = len(ys)
drivable_pct = (gt_seg==0).mean()*100
crossing_pct = (gt_seg==1).mean()*100
walkway_pct  = (gt_seg==2).mean()*100

print(f"Detections: {n_det} | Heatmap max: {hm_max:.4f}")

print("Building figure...")
fig = plt.figure(figsize=(26, 20), facecolor=BG)
gs  = gridspec.GridSpec(4, 6, figure=fig,
                         hspace=0.42, wspace=0.30,
                         left=0.04, right=0.97,
                         top=0.92,  bottom=0.05)

fig.text(0.5, 0.965,
         "NEURAL DRIVE  —  Multi-Task Autonomous Perception  [T4 · 24 Epochs]",
         ha="center", color="#00e5ff", fontsize=17, fontweight="bold",
         fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor=BG,
                   edgecolor="#00e5ff", linewidth=1.5))
fig.text(0.5, 0.942,
         "LiDAR+Camera Fusion  |  nuScenes Mini  |  128ch  |  BEV 200×200  |  Real Detection Targets",
         ha="center", color="#3a6070", fontsize=9, fontfamily="monospace")

def style(ax, title, border="#333"):
    ax.set_facecolor(PBG)
    ax.set_title(title, color="white", fontweight="bold",
                 fontsize=9, pad=6, fontfamily="monospace")
    ax.set_xlabel("X (m)", color="#555", fontsize=7)
    ax.set_ylabel("Y (m)", color="#555", fontsize=7)
    ax.tick_params(colors="#444", labelsize=6)
    ax.grid(True, alpha=0.07, color="white", zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor(border); sp.set_linewidth(1.5)

cam_names  = ["FRONT","FRONT LEFT","FRONT RIGHT","BACK","BACK LEFT","BACK RIGHT"]
cam_colors = ["#00e5ff"]*3 + ["#ff6b35"]*3
for i in range(6):
    ax = fig.add_subplot(gs[0, i])
    img = np.clip(sample["images"][i].permute(1,2,0).numpy()*std_img+mean_img, 0, 1)
    ax.imshow(img)
    ax.set_title(cam_names[i], color=cam_colors[i],
                 fontsize=7, fontweight="bold", fontfamily="monospace", pad=3)
    ax.axis("off")
    for sp in ax.spines.values():
        sp.set_edgecolor(cam_colors[i]); sp.set_linewidth(1.2); sp.set_visible(True)

ax1 = fig.add_subplot(gs[1, 0:2])
im  = ax1.imshow(hm_norm, cmap="hot", vmin=0, vmax=1,
                 extent=[-50,50,-50,50], origin="lower", aspect="auto")
style(ax1, "◈  DETECTION HEATMAP — 200×200 BEV", "#a855f7")
cb = plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
cb.ax.tick_params(colors="#555", labelsize=6)
ax1.plot(0, 0, "^", color="#00e5ff", ms=10, zorder=10, label="Ego")
for y, x in zip(ys, xs):
    ax1.add_patch(plt.Circle(
        (-50+(x/W_bev)*100, -50+(y/H_bev)*100),
        1.0, color="#ff6b35", alpha=0.8, fill=False, lw=1.5, zorder=8))
ax1.legend(facecolor=PBG, labelcolor="white", fontsize=7, edgecolor="#333")
ax1.text(0, -48, f"{n_det} detections | max={hm_max:.4f}",
         color="#ff6b35", fontsize=7, fontfamily="monospace", ha="center", va="bottom")

ax2 = fig.add_subplot(gs[1, 2:4])
ax2.imshow(seg_to_rgb(gt_seg), extent=[-50,50,-50,50], origin="lower", aspect="auto")
style(ax2, "◈  HD MAP + DETECTION OVERLAY", "#00ff88")
for y, x in zip(ys, xs):
    ax2.plot(-50+(x/W_bev)*100, -50+(y/H_bev)*100,
             "o", color="#ff6b35", ms=4, alpha=0.9, zorder=8, mec="white", mew=0.3)
ax2.plot(0, 0, "^", color="white", ms=12, zorder=10)
unique_cls = np.unique(gt_seg)
patches = [mpatches.Patch(facecolor=SEG_COLORS[c], label=SEG_NAMES[c], edgecolor="none")
           for c in unique_cls if c < 7]
patches += [mpatches.Patch(facecolor="#ff6b35", label=f"Detections ({n_det})",
                            edgecolor="white", lw=0.5)]
ax2.legend(handles=patches, facecolor=PBG, labelcolor="white",
           fontsize=6, loc="upper left", framealpha=0.92, edgecolor="#333")
ax2.text(46, -46,
         f"Drivable: {drivable_pct:.0f}%\nCrossing: {crossing_pct:.0f}%\nWalkway: {walkway_pct:.0f}%",
         color="white", fontsize=6.5, fontfamily="monospace", va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor=PBG, edgecolor="#444", alpha=0.92))

ax3 = fig.add_subplot(gs[1, 4:6])
seg_probs = preds["segmentation"][0].softmax(0).cpu().float().numpy()
pred_rgb  = np.zeros((seg_probs.shape[1], seg_probs.shape[2], 3), dtype=np.float32)
for c in range(14): pred_rgb += seg_probs[c,:,:,np.newaxis] * SEG_COLORS[c]
pred_rgb  = pred_rgb.clip(0, 1)
ax3.imshow(pred_rgb, extent=[-50,50,-50,50], origin="lower", aspect="auto")
style(ax3, "◈  PREDICTED SEGMENTATION", "#ff6b35")
ax3.plot(0, 0, "^", color="white", ms=12, zorder=10)
seg_am = seg_probs.argmax(0)
pred_patches = [
    mpatches.Patch(facecolor=SEG_COLORS[c],
                   label=f"{SEG_NAMES[c]} ({(seg_am==c).mean()*100:.0f}%)",
                   edgecolor="none")
    for c in np.unique(seg_am) if c < 8
]
if pred_patches:
    ax3.legend(handles=pred_patches, facecolor=PBG, labelcolor="white",
               fontsize=6, loc="upper right", framealpha=0.92, edgecolor="#333")

ax4 = fig.add_subplot(gs[2, 0:2])
ax4.imshow(pca, extent=[-50,50,-50,50], origin="lower", aspect="auto")
style(ax4, "◈  FUSED BEV FEATURES (PCA)", "#00d4ff")
ax4.plot(0, 0, "^", color="white", ms=12, zorder=10)
ax4.text(-46, 90,
         f"LiDAR: {sample['points'].shape[0]:,} pts\nBEV: 200×200 @ 0.5m/cell\nChannels: 128",
         color="#aaa", fontsize=6.5, fontfamily="monospace", va="top",
         bbox=dict(boxstyle="round,pad=0.3", facecolor=PBG, edgecolor="#444", alpha=0.9))

ax5 = fig.add_subplot(gs[2, 2:4])
gt_dist   = [(gt_seg==i).mean() for i in range(8)]
pred_dist = [float((seg_am==i).mean()) for i in range(8)]
x = np.arange(8); w = 0.35
ax5.bar(x-w/2, gt_dist,   w, label="GT",   color="#00ff88", alpha=0.8)
ax5.bar(x+w/2, pred_dist, w, label="Pred", color="#ff6b35", alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(["Drive","PedX","Walk","Stop","Park","RDiv","LDiv","BG"],
                     color="#aaa", fontsize=7, rotation=30)
ax5.set_facecolor(PBG); ax5.tick_params(colors="#444")
ax5.set_title("◈  GT vs Predicted Distribution", color="white",
               fontweight="bold", fontsize=9, fontfamily="monospace", pad=6)
ax5.legend(facecolor=PBG, labelcolor="white", fontsize=8, edgecolor="#333")
ax5.grid(True, alpha=0.08, axis="y", color="white")
for sp in ax5.spines.values(): sp.set_edgecolor("#333")

ax6 = fig.add_subplot(gs[2, 4:6])
cp = seg_probs[:8].mean(axis=(1,2))
ax6.barh(range(8), cp, color=[SEG_COLORS[i] for i in range(8)],
         edgecolor="none", height=0.6)
ax6.set_yticks(range(8))
ax6.set_yticklabels(SEG_NAMES[:8], color="#aaa", fontsize=8)
ax6.set_xlabel("Mean Probability", color="#555", fontsize=8)
ax6.set_title("◈  Seg Class Confidence", color="white",
               fontweight="bold", fontsize=9, fontfamily="monospace", pad=6)
ax6.set_facecolor(PBG); ax6.tick_params(colors="#444")
for sp in ax6.spines.values(): sp.set_edgecolor("#333")
for i, v in enumerate(cp):
    ax6.text(v+0.001, i, f"{v:.3f}", color="white", va="center", fontsize=7)

ax_l = fig.add_subplot(gs[3, :])
train_l = [19.6169,5.4164,3.7177,6.9460,6.3279,5.5748,4.8918,4.3142,3.9149,
           3.5893,3.2831,3.0854,2.9243,2.7638,2.6469,2.5622,2.4545,2.3753,
           2.3039,2.2618,2.1914,2.1853,2.1536,2.1361,2.1086]
val_pts = [10.4325,29.5809,None,7.3492,None,None,10.3596,None,None,
           4.6556,None,None,3.0545,None,None,2.7334,None,None,
           2.4927,None,None,2.3426,None,None,2.3056]
ep = list(range(1, len(train_l)+1))
ax_l.fill_between(ep, train_l, alpha=0.18, color="#00d4ff")
ax_l.plot(ep, train_l, color="#00e5ff", lw=2, marker="o", ms=3, label="Train Loss")
val_ep = [e for e,v in zip(ep,val_pts) if v is not None]
val_v  = [v for v in val_pts if v is not None]
ax_l.plot(val_ep, val_v, color="#00ff88", lw=2, marker="s", ms=5,
          label="Val Loss", linestyle="--")
ax_l.set_facecolor(PBG)
ax_l.set_title("◈  TRAINING HISTORY — 24 Epochs | Real Detection Targets Fixed",
               color="white", fontweight="bold", fontsize=10, fontfamily="monospace", pad=6)
ax_l.set_xlabel("Epoch", color="#555", fontsize=8)
ax_l.set_ylabel("MTL Loss", color="#555", fontsize=8)
ax_l.tick_params(colors="#444")
ax_l.legend(facecolor=PBG, labelcolor="white", fontsize=9, edgecolor="#333")
ax_l.grid(True, alpha=0.08, color="white")
ax_l.set_xlim(1, 24); ax_l.set_ylim(0, max(train_l)*1.08)
for sp in ax_l.spines.values(): sp.set_edgecolor("#1a3040")
ax_l.annotate(f"Final: {train_l[-1]:.4f}",
              xy=(24, train_l[-1]), xytext=(20, 5),
              color="#00ffaa", fontsize=9,
              arrowprops=dict(arrowstyle="->", color="#00ffaa", lw=1.5))

cards = [
    ("35.7M",                              "TOTAL PARAMS",  "#00e5ff", gs[3,0:2]),
    (f"{sample['points'].shape[0]//1000}K","LIDAR POINTS",  "#00ff88", gs[3,2:3]),
    (f"{n_det}",                           "DETECTIONS",    "#a855f7", gs[3,3:4]),
    (f"{train_l[-1]:.4f}",                 "FINAL LOSS",    "#ff6b35", gs[3,4:6]),
]

# Override loss curve with metric cards for bottom row
ax_l.remove()
for val, lbl, col, gsloc in cards:
    ax_c = fig.add_subplot(gsloc)
    ax_c.set_facecolor(PBG); ax_c.axis("off")
    ax_c.text(0.5, 0.60, val, color=col, fontsize=30, fontweight="bold",
              fontfamily="monospace", ha="center", va="center", transform=ax_c.transAxes)
    ax_c.text(0.5, 0.20, lbl, color="#3a6070", fontsize=8,
              fontfamily="monospace", ha="center", va="center", transform=ax_c.transAxes)
    for sp in ax_c.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(1.8); sp.set_visible(True)

# Re-add loss curve
ax_l = fig.add_subplot(gs[3, :])
ax_l.fill_between(ep, train_l, alpha=0.18, color="#00d4ff")
ax_l.plot(ep, train_l, color="#00e5ff", lw=2, marker="o", ms=3, label="Train Loss")
ax_l.plot(val_ep, val_v, color="#00ff88", lw=2, marker="s", ms=5,
          label="Val Loss", linestyle="--")
ax_l.set_facecolor(PBG)
ax_l.set_title("◈  TRAINING HISTORY — 24 Epochs | Real Detection Targets Fixed",
               color="white", fontweight="bold", fontsize=10, fontfamily="monospace", pad=6)
ax_l.set_xlabel("Epoch", color="#555", fontsize=8)
ax_l.set_ylabel("MTL Loss", color="#555", fontsize=8)
ax_l.tick_params(colors="#444")
ax_l.legend(facecolor=PBG, labelcolor="white", fontsize=9, edgecolor="#333")
ax_l.grid(True, alpha=0.08, color="white")
ax_l.set_xlim(1, 24); ax_l.set_ylim(0, max(train_l)*1.08)
for sp in ax_l.spines.values(): sp.set_edgecolor("#1a3040")
ax_l.annotate(f"Final: {train_l[-1]:.4f}",
              xy=(24, train_l[-1]), xytext=(20, 5),
              color="#00ffaa", fontsize=9,
              arrowprops=dict(arrowstyle="->", color="#00ffaa", lw=1.5))

plt.savefig("inference_results.png", dpi=160, bbox_inches="tight", facecolor=BG)
print("✅ Saved → inference_results.png")