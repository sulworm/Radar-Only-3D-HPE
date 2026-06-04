"""Smoke-test MM-Fi radar-only splits and labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.adapters.mmfi import (
    describe,
    protocol_actions,
    read_ground_truth,
    read_mmwave_frame,
    split_sequences,
    streaming_mean_pose,
    validate_sequences,
)


def _constant_pose_metrics(sequences, mean_pose):
    total_distance = 0.0
    total_root_distance = 0.0
    total_joints = 0
    pck_hits = {0.05: 0, 0.10: 0, 0.15: 0}

    for sequence in sequences:
        gt = read_ground_truth(sequence)
        pred = np.broadcast_to(mean_pose, gt.shape)
        distances = np.linalg.norm(pred - gt, axis=-1)
        root_pred = pred - pred[:, :1, :]
        root_gt = gt - gt[:, :1, :]
        root_distances = np.linalg.norm(root_pred - root_gt, axis=-1)

        total_distance += float(distances.sum())
        total_root_distance += float(root_distances.sum())
        total_joints += int(distances.size)
        for threshold in pck_hits:
            pck_hits[threshold] += int(np.count_nonzero(distances <= threshold))

    if total_joints == 0:
        raise ValueError("No joints evaluated")
    return {
        "mpjpe_mm": total_distance / total_joints * 1000.0,
        "root_relative_mpjpe_mm": total_root_distance / total_joints * 1000.0,
        "pck@50mm": pck_hits[0.05] / total_joints,
        "pck@100mm": pck_hits[0.10] / total_joints,
        "pck@150mm": pck_hits[0.15] / total_joints,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Path to MMFI_mmwave. Defaults to .benchmark/MM-Fi/MMFI_mmwave.")
    parser.add_argument("--protocol", default="p3", help="p1, p2, or p3.")
    parser.add_argument("--setting", default="s2", choices=("s1", "s2", "s3"))
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--random_ratio", type=float, default=0.8)
    parser.add_argument("--test_scene", default="E04")
    parser.add_argument("--check_frames", action="store_true", help="Check every frame file exists.")
    parser.add_argument("--mean_pose_baseline", action="store_true", help="Evaluate train mean pose on the chosen split.")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    dataset_info = describe(args.root)
    train_sequences = split_sequences(
        root=args.root,
        protocol=args.protocol,
        setting=args.setting,
        split="train",
        random_seed=args.random_seed,
        random_ratio=args.random_ratio,
        test_scene=args.test_scene,
    )
    target_sequences = split_sequences(
        root=args.root,
        protocol=args.protocol,
        setting=args.setting,
        split=args.split,
        random_seed=args.random_seed,
        random_ratio=args.random_ratio,
        test_scene=args.test_scene,
    )
    validation = validate_sequences(target_sequences, check_frames=args.check_frames)

    print("MM-Fi dataset summary:")
    print(dataset_info)
    print(
        f"Protocol {args.protocol}: {len(protocol_actions(args.protocol))} actions | "
        f"setting={args.setting} split={args.split}"
    )
    print(f"Train sequences: {len(train_sequences)} ({len(train_sequences) * 297} frames)")
    print(f"Target sequences: {len(target_sequences)} ({len(target_sequences) * 297} frames)")
    print(f"Validation: {validation}")

    sample = target_sequences[0]
    sample_pose = read_ground_truth(sample)
    sample_frame = read_mmwave_frame(sample, 1)
    print(
        "Sample: "
        f"{sample.scene}/{sample.subject}/{sample.action} "
        f"pose={sample_pose.shape} mmwave_frame001={sample_frame.shape}"
    )

    result = {
        "dataset": dataset_info,
        "protocol": args.protocol,
        "setting": args.setting,
        "split": args.split,
        "train_sequences": len(train_sequences),
        "target_sequences": len(target_sequences),
        "validation": validation,
        "sample": {
            "scene": sample.scene,
            "subject": sample.subject,
            "action": sample.action,
            "pose_shape": tuple(sample_pose.shape),
            "mmwave_frame001_shape": tuple(sample_frame.shape),
        },
    }

    if args.mean_pose_baseline:
        mean_pose = streaming_mean_pose(train_sequences)
        metrics = _constant_pose_metrics(target_sequences, mean_pose)
        result["mean_pose_baseline"] = metrics
        print("Mean-pose baseline:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
