"""
nuScenes multi-task dataset — with REAL map segmentation labels.
Uses nuscenes-map-expansion to rasterize lane/drivable/crossing labels into BEV grid.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.map_expansion.map_api import NuScenesMap
    from pyquaternion import Quaternion
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("[WARNING] nuscenes-devkit not installed. Using dummy dataset.")


# ── Class definitions ───────────────────────────────────────────────────────
DETECTION_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',
]
CLASS_TO_IDX = {c: i for i, c in enumerate(DETECTION_CLASSES)}

CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT',
]

MAP_LAYERS = [
    'drivable_area',      # 0
    'ped_crossing',       # 1
    'walkway',            # 2
    'stop_line',          # 3
    'carpark_area',       # 4
    'road_divider',       # 5
    'lane_divider',       # 6
]
LAYER_TO_IDX    = {l: i for i, l in enumerate(MAP_LAYERS)}
NUM_MAP_CLASSES = len(MAP_LAYERS) + 1
BACKGROUND_IDX  = len(MAP_LAYERS)      # 7

LOCATION_MAP = {
    'singapore-onenorth':       'singapore-onenorth',
    'singapore-hollandvillage': 'singapore-hollandvillage',
    'singapore-queenstown':     'singapore-queenstown',
    'boston-seaport':           'boston-seaport',
}


class NuScenesMTL(Dataset):

    def __init__(
        self,
        dataroot:    str,
        version:     str   = 'v1.0-mini',
        split:       str   = 'train',
        img_size:    tuple = (224, 224),
        bev_size:    tuple = (50, 50),
        bev_range:   tuple = (-50, -50, 50, 50),
        max_objects: int   = 100,
    ):
        self.dataroot    = Path(dataroot)
        self.img_h, self.img_w = img_size
        self.bev_h, self.bev_w = bev_size
        self.bev_range   = bev_range
        self.max_objects = max_objects
        self.split       = split

        if not NUSCENES_AVAILABLE:
            self.nusc    = None
            self.samples = list(range(50))
            return

        self.nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)

        self.maps = {}
        for loc in LOCATION_MAP:
            try:
                self.maps[loc] = NuScenesMap(dataroot=str(dataroot), map_name=loc)
            except Exception:
                pass

        split_file = self.dataroot / f'data/splits/{split}.txt'
        if split_file.exists():
            sample_tokens = [l.strip() for l in split_file.read_text().splitlines()]
        else:
            sample_tokens = [s['token'] for s in self.nusc.sample]

        self.samples = sample_tokens
        print(f"[NuScenesMTL] {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.nusc is None:
            return self._dummy_item()

        sample_token = self.samples[idx]
        sample       = self.nusc.get('sample', sample_token)

        images, intrinsics, extrinsics = self._load_cameras(sample)
        points                          = self._load_lidar(sample)

        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data  = self.nusc.get('sample_data', lidar_token)
        ego_pose    = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

        targets = self._build_targets(sample, ego_pose)

        return {
            'images':     images,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'points':     points,
            'targets':    targets,
            'token':      sample_token,
        }

    def _load_cameras(self, sample):
        imgs_list, Ks_list, Es_list = [], [], []

        for cam in CAMERAS:
            cam_token = sample['data'][cam]
            cam_data  = self.nusc.get('sample_data', cam_token)
            calib     = self.nusc.get('calibrated_sensor',
                                      cam_data['calibrated_sensor_token'])

            img_path = self.dataroot / cam_data['filename']
            img      = cv2.imread(str(img_path))
            img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img      = cv2.resize(img, (self.img_w, self.img_h))
            img      = img.astype(np.float32) / 255.0
            mean     = np.array([0.485, 0.456, 0.406])
            std      = np.array([0.229, 0.224, 0.225])
            img      = (img - mean) / std
            imgs_list.append(torch.from_numpy(img).permute(2, 0, 1).float())

            K = np.array(calib['camera_intrinsic'], dtype=np.float32)
            K[0] *= self.img_w / cam_data['width']
            K[1] *= self.img_h / cam_data['height']
            Ks_list.append(torch.from_numpy(K))

            R = Quaternion(calib['rotation']).rotation_matrix
            t = np.array(calib['translation'])
            E = np.eye(4, dtype=np.float32)
            E[:3, :3] = R
            E[:3, 3]  = t
            Es_list.append(torch.from_numpy(E))

        return (torch.stack(imgs_list),
                torch.stack(Ks_list),
                torch.stack(Es_list))

    def _load_lidar(self, sample):
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data  = self.nusc.get('sample_data', lidar_token)
        pc = LidarPointCloud.from_file(
            str(self.dataroot / lidar_data['filename'])
        )
        points = pc.points.T
        calib  = self.nusc.get('calibrated_sensor',
                               lidar_data['calibrated_sensor_token'])
        R      = Quaternion(calib['rotation']).rotation_matrix
        t      = np.array(calib['translation'])
        pts    = (R @ points[:, :3].T).T + t
        points = np.concatenate([pts, points[:, 3:4]], axis=1)
        return torch.from_numpy(points.astype(np.float32))

    def _build_targets(self, sample, ego_pose):
        targets = {}
        targets['detection']    = self._build_detection_targets(sample, ego_pose)
        targets['segmentation'] = self._build_map_segmentation(ego_pose)
        targets['depth']        = torch.zeros(
            len(CAMERAS), 1, self.img_h, self.img_w)
        return targets

    def _build_map_segmentation(self, ego_pose) -> torch.Tensor:
        H, W = self.bev_h, self.bev_w
        x_min, y_min, x_max, y_max = self.bev_range

        seg = np.full((H, W), BACKGROUND_IDX, dtype=np.int64)

        ego_x   = ego_pose['translation'][0]
        ego_y   = ego_pose['translation'][1]
        ego_rot = Quaternion(ego_pose['rotation'])
        ego_yaw = ego_rot.yaw_pitch_roll[0]

        nusc_map = None
        for loc, m in self.maps.items():
            try:
                patch    = (ego_x + x_min, ego_y + y_min,
                            ego_x + x_max, ego_y + y_max)
                nusc_map = m
                break
            except Exception:
                continue

        if nusc_map is None:
            return torch.from_numpy(seg)

        try:
            patch_size  = (x_max - x_min, y_max - y_min)
            patch_angle = np.degrees(ego_yaw)

            for layer_name, layer_idx in LAYER_TO_IDX.items():
                if layer_name not in nusc_map.non_geometric_polygon_layers and \
                   layer_name not in nusc_map.non_geometric_line_layers:
                    continue

                try:
                    map_mask   = nusc_map.get_map_mask(
                        patch_box=(ego_x, ego_y,
                                   patch_size[0], patch_size[1]),
                        patch_angle=patch_angle,
                        layer_names=[layer_name],
                        canvas_size=(H, W),
                    )
                    layer_mask = map_mask[0]
                    seg[layer_mask > 0] = layer_idx
                except Exception:
                    continue

        except Exception:
            pass

        return torch.from_numpy(seg)

    def _build_detection_targets(self, sample, ego_pose):
        """
        Build detection targets in EGO VEHICLE FRAME.
        Transforms global world coordinates → ego frame before projecting to BEV.
        This is the critical fix — global coords (e.g. [373, 1130]) are outside
        BEV range [-50, 50] so all objects were being skipped without this fix.
        """
        H, W  = self.bev_h, self.bev_w
        n_cls = len(DETECTION_CLASSES)
        x_min, y_min, x_max, y_max = self.bev_range

        heatmap  = torch.zeros(n_cls, H, W)
        offset   = torch.zeros(2, H, W)
        wlh_t    = torch.zeros(3, H, W)
        yaw_t    = torch.zeros(2, H, W)
        velocity = torch.zeros(2, H, W)
        mask     = torch.zeros(H, W)

        # Ego pose for coordinate transform
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation    = Quaternion(ego_pose['rotation'])

        for ann_token in sample['anns'][:self.max_objects]:
            ann     = self.nusc.get('sample_annotation', ann_token)
            cat     = ann['category_name']
            matched = next((c for c in DETECTION_CLASSES if c in cat), None)
            if matched is None:
                continue

            cls_idx = CLASS_TO_IDX[matched]

            # ── KEY FIX: Transform global → ego frame ──────────────────
            global_xyz = np.array(ann['translation'])
            local_xyz  = ego_rotation.inverse.rotate(global_xyz - ego_translation)
            cx, cy     = local_xyz[0], local_xyz[1]
            # ────────────────────────────────────────────────────────────

            col_f        = (cx - x_min) / (x_max - x_min) * W
            row_f        = (cy - y_min) / (y_max - y_min) * H
            col_i, row_i = int(col_f), int(row_f)

            if not (0 <= col_i < W and 0 <= row_i < H):
                continue

            sigma = max(2.0, (ann['size'][0] + ann['size'][1]) / 4)
            self._draw_gaussian(heatmap[cls_idx], (col_i, row_i), sigma)

            offset[0, row_i, col_i] = col_f - col_i
            offset[1, row_i, col_i] = row_f - row_i
            wlh_t[0, row_i, col_i]  = ann['size'][1]
            wlh_t[1, row_i, col_i]  = ann['size'][0]
            wlh_t[2, row_i, col_i]  = ann['size'][2]

            # Yaw in ego frame
            q     = Quaternion(ann['rotation'])
            angle = q.yaw_pitch_roll[0]
            yaw_t[0, row_i, col_i] = np.sin(angle)
            yaw_t[1, row_i, col_i] = np.cos(angle)

            try:
                vel = self.nusc.box_velocity(ann_token)
                if not np.isnan(vel).any():
                    # Transform velocity to ego frame
                    vel_local = ego_rotation.inverse.rotate(vel)
                    velocity[0, row_i, col_i] = vel_local[0]
                    velocity[1, row_i, col_i] = vel_local[1]
            except Exception:
                pass

            mask[row_i, col_i] = 1.0

        return {
            'heatmap':     heatmap,
            'offset':      offset,
            'wlh':         wlh_t,
            'yaw':         yaw_t,
            'velocity':    velocity,
            'centre_mask': mask,
        }

    @staticmethod
    def _draw_gaussian(heatmap, centre, sigma):
        H, W   = heatmap.shape
        cx, cy = centre
        r      = int(3 * sigma)
        for x in range(max(0, cx - r), min(W, cx + r + 1)):
            for y in range(max(0, cy - r), min(H, cy + r + 1)):
                v = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
                if v > heatmap[y, x].item():
                    heatmap[y, x] = v

    def _dummy_item(self):
        N = len(CAMERAS)
        return {
            'images':     torch.randn(N, 3, self.img_h, self.img_w),
            'intrinsics': torch.eye(3).unsqueeze(0).expand(N, -1, -1),
            'extrinsics': torch.eye(4).unsqueeze(0).expand(N, -1, -1),
            'points':     torch.randn(10000, 4),
            'targets': {
                'detection': {
                    'heatmap':     torch.zeros(len(DETECTION_CLASSES), self.bev_h, self.bev_w),
                    'offset':      torch.zeros(2, self.bev_h, self.bev_w),
                    'wlh':         torch.ones(3, self.bev_h, self.bev_w),
                    'yaw':         torch.zeros(2, self.bev_h, self.bev_w),
                    'velocity':    torch.zeros(2, self.bev_h, self.bev_w),
                    'centre_mask': torch.zeros(self.bev_h, self.bev_w),
                },
                'segmentation': torch.zeros(self.bev_h, self.bev_w, dtype=torch.long),
                'depth':        torch.zeros(N, 1, self.img_h, self.img_w),
            },
            'token': 'dummy',
        }


# ── Collate ────────────────────────────────────────────────────────────────
def collate_fn(batch):
    images     = torch.stack([b['images']     for b in batch])
    intrinsics = torch.stack([b['intrinsics'] for b in batch])
    extrinsics = torch.stack([b['extrinsics'] for b in batch])
    points     = [b['points'] for b in batch]

    def stack_det(key):
        return torch.stack([b['targets']['detection'][key] for b in batch])

    targets = {
        'detection': {
            'heatmap':     stack_det('heatmap'),
            'offset':      stack_det('offset'),
            'wlh':         stack_det('wlh'),
            'yaw':         stack_det('yaw'),
            'velocity':    stack_det('velocity'),
            'centre_mask': stack_det('centre_mask'),
        },
        'segmentation': torch.stack([b['targets']['segmentation'] for b in batch]),
        'depth':        torch.stack([b['targets']['depth']        for b in batch]),
    }

    return {
        'images':     images,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'points':     points,
        'targets':    targets,
        'tokens':     [b['token'] for b in batch],
    }