"""Smoke-test mRI data loading and official mmWave result summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.adapters.mri import (
    aligned_feature_pose,
    all_official_summaries,
    describe,
    protocol_actions,
    read_official_results,
    read_pose_label_dict,
    read_radar_featuremap,
    summarize_official_results,
    subject_item,
)
from benchmarks.metrics import mpjpe, pa_mpjpe


def _mean_pose_smoke(pose: np.ndarray) -> dict[str, float]:
    if pose.size == 0:
        raise ValueError("No pose frames available for smoke metrics")
    mean_pose = pose.mean(axis=0, keepdims=True)
    pred = np.broadcast_to(mean_pose, pose.shape)
    return {
        "mean_pose_mpjpe_mm": mpjpe(pred, pose) * 1000.0,
        "mean_pose_pa_mpjpe_mm": pa_mpjpe(pred, pose, scale=True) * 1000.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Path to mRI dataset_release.")
    parser.add_argument("--subject", default="subject1")
    parser.add_argument("--protocol", default="p2", help="p1 or p2.")
    parser.add_argument("--split", default="all", choices=("all", "random", "subject"))
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    dataset_info = describe(args.root)
    item = subject_item(args.subject, root=args.root)
    label_data = read_pose_label_dict(item)
    featuremap = read_radar_featuremap(item)
    features, pose, frames = aligned_feature_pose(item, protocol=args.protocol)
    mean_pose_metrics = _mean_pose_smoke(pose)

    result = {
        "dataset": dataset_info,
        "sample": {
            "subject": item.subject,
            "pose_keys": sorted(label_data.keys()),
            "refined_gt_shape_raw": tuple(np.asarray(label_data["refined_gt_kps"]).shape),
            "radar_featuremap_shape": tuple(featuremap.shape),
            "protocol": args.protocol,
            "protocol_actions": protocol_actions(args.protocol),
            "protocol_feature_shape": tuple(features.shape),
            "protocol_pose_shape": tuple(pose.shape),
            "protocol_frame_range": [int(frames[0]), int(frames[-1])] if len(frames) else [],
            "mean_pose_smoke": mean_pose_metrics,
        },
        "official_results": {},
    }

    print("mRI dataset summary:")
    print(dataset_info)
    print(
        f"Sample {item.subject}: featuremap={featuremap.shape}, "
        f"raw refined_gt={np.asarray(label_data['refined_gt_kps']).shape}, "
        f"{args.protocol} frames={pose.shape[0]}"
    )
    print("Mean-pose smoke on sample protocol slice:")
    for key, value in mean_pose_metrics.items():
        print(f"  {key}: {value:.4f}")

    if args.split == "all":
        official = all_official_summaries(args.root)
    else:
        official = {}
        for protocol in ("p1", "p2"):
            array = read_official_results(args.root, split=args.split, protocol=protocol)
            official[f"{args.split}_{protocol}"] = summarize_official_results(array)

    result["official_results"] = official
    print("Official mmWave results:")
    for name, summary in official.items():
        mean = summary["mean"]
        std = summary["std"]
        mean_diff = max(summary["mean_abs_diff"])
        std_diff = max(summary["std_abs_diff"])
        print(
            f"  {name}: MPJPE={mean[0]:.2f}+/-{std[0]:.2f} mm, "
            f"PA={mean[1]:.2f}+/-{std[1]:.2f} mm, "
            f"mean_diff={mean_diff:.2e}, std_diff={std_diff:.2e}"
        )

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
