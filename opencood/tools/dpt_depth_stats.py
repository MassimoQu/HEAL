import argparse
import sys
from pathlib import Path
import random

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, "/home/qqxluca/vggt_series_4_coop/Depth-Anything-3/src")

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.io.input_processor import InputProcessor

from opencood.utils.pcd_utils import read_pcd, pcd_to_np
from opencood.utils.transformation_utils import x1_to_x2


UE4_TO_OPENCV = np.array(
    [[0, 0, 1, 0],
     [1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, 0, 1]],
    dtype=np.float32,
)


def _collect_samples(root_dir: Path, cam_id: int, max_samples: int):
    samples = []
    scenarios = [p for p in root_dir.iterdir() if p.is_dir()]
    scenarios.sort()
    for scenario in scenarios:
        cav_dirs = [p for p in scenario.iterdir() if p.is_dir()]
        cav_dirs.sort()
        for cav in cav_dirs:
            yaml_files = sorted(cav.glob("*.yaml"))
            for ypath in yaml_files:
                ts = ypath.stem
                img_path = cav / f"{ts}_camera{cam_id}.png"
                pcd_path = cav / f"{ts}.pcd"
                if img_path.exists() and pcd_path.exists():
                    samples.append((ypath, img_path, pcd_path))
                    break
            if len(samples) >= max_samples:
                return samples
    return samples


def _project_lidar_to_camera(pcd_xyz, cam_to_lidar, K, img_h, img_w):
    pcd_xyz = np.asarray(pcd_xyz)
    if pcd_xyz.ndim == 1:
        pcd_xyz = pcd_xyz.reshape(1, -1)
    lidar_to_cam = np.linalg.inv(cam_to_lidar)
    pts = np.concatenate([pcd_xyz[:, :3], np.ones((pcd_xyz.shape[0], 1), dtype=np.float32)], axis=1)
    pts_cam = (lidar_to_cam @ pts.T).T
    if pts_cam.ndim == 1:
        pts_cam = pts_cam.reshape(1, -1)
    if pts_cam.ndim != 2 or pts_cam.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    depth = pts_cam[:, 2]
    mask = depth > 0.1
    pts_cam = pts_cam[mask]
    depth = depth[mask]
    if pts_cam.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    uv = (K @ pts_cam[:, :3].T).T
    if uv.ndim != 2 or uv.shape[1] < 3:
        return np.array([]), np.array([]), np.array([])
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return u[mask], v[mask], depth[mask]


def _stats(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="dataset/OPV2V/train")
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--process_method", type=str, default="upper_bound_resize")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    random.seed(args.seed)

    samples = _collect_samples(root_dir, args.cam_id, args.num_samples)
    if not samples:
        print("No samples found under", root_dir)
        return

    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
    model.eval()
    processor = InputProcessor()

    all_dpt = []
    all_lidar = []
    all_ratio = []

    for idx, (yaml_path, img_path, pcd_path) in enumerate(samples):
        with open(yaml_path, "r") as f:
            params = yaml.safe_load(f)

        cam_key = f"camera{args.cam_id}"
        if cam_key not in params:
            continue

        cam_coords = np.array(params[cam_key]["cords"], dtype=np.float32)
        lidar_pose = np.array(params.get("lidar_pose_clean", params.get("lidar_pose")), dtype=np.float32)
        if lidar_pose is None:
            continue
        cam_to_lidar = x1_to_x2(cam_coords, lidar_pose).astype(np.float32)
        cam_to_lidar = cam_to_lidar @ UE4_TO_OPENCV
        K = np.array(params[cam_key]["intrinsic"], dtype=np.float32)

        img = Image.open(img_path).convert("RGB")
        img_tensor, _, ixts = processor(
            [img],
            intrinsics=[K],
            process_res=args.process_res,
            process_res_method=args.process_method,
            num_workers=1,
            sequential=True,
        )

        img_tensor = img_tensor.to(args.device)
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.unsqueeze(0)
        if next(model.model.parameters()).device != img_tensor.device:
            model.model.to(img_tensor.device)
        with torch.no_grad():
            raw = model.model(
                img_tensor,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=False,
                ref_view_strategy="saddle_balanced",
            )
        depth = raw["depth"]
        if depth.dim() == 5:
            depth = depth.squeeze(2)
        depth = depth[0, 0].detach().cpu().numpy()
        img_h, img_w = depth.shape

        if ixts is not None:
            if ixts.ndim == 4:
                K_proc = ixts[0, 0].numpy()
            else:
                K_proc = ixts[0].numpy()
        else:
            K_proc = K

        try:
            pcd, _ = read_pcd(str(pcd_path))
        except Exception:
            pcd = pcd_to_np(str(pcd_path))
        u, v, d_lidar = _project_lidar_to_camera(pcd, cam_to_lidar, K_proc, img_h, img_w)
        if d_lidar.size == 0:
            continue

        u_int = np.clip(np.round(u).astype(np.int32), 0, img_w - 1)
        v_int = np.clip(np.round(v).astype(np.int32), 0, img_h - 1)
        dpt_samples = depth[v_int, u_int]

        all_lidar.append(d_lidar)
        all_dpt.append(dpt_samples)
        ratio = dpt_samples / np.maximum(d_lidar, 1e-6)
        all_ratio.append(ratio)

        print(f"[{idx+1}/{len(samples)}] {img_path.name} lidar_pts={d_lidar.size} "
              f"dpt_med={np.median(dpt_samples):.3f} lidar_med={np.median(d_lidar):.3f}")

    all_dpt = np.concatenate(all_dpt) if all_dpt else np.array([])
    all_lidar = np.concatenate(all_lidar) if all_lidar else np.array([])
    all_ratio = np.concatenate(all_ratio) if all_ratio else np.array([])

    print("\nDPT depth stats:", _stats(all_dpt))
    print("LiDAR depth stats:", _stats(all_lidar))
    print("DPT/LiDAR ratio stats:", _stats(all_ratio))


if __name__ == "__main__":
    main()
