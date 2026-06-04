"""Smoke-test MARS loading and MARS-style localization metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.adapters.mars import MARS_JOINT_NAMES, describe, load_split, mars_label_to_pose
from benchmarks.metrics import mars_localization_table, mpjpe


def _load_prediction(path: Path):
    pred = np.load(path)
    if pred.ndim == 2 and pred.shape[1] == 57:
        return mars_label_to_pose(pred)
    if pred.ndim == 3 and pred.shape[1:] == (19, 3):
        return pred.astype(np.float32)
    raise ValueError(f"Unsupported prediction shape {pred.shape}; expected [N,57] or [N,19,3]")


def _print_mars_summary(rows):
    avg = rows[-1]
    print(
        "MARS-style average (cm): "
        f"x MAE/RMSE={avg['x_mae']:.2f}/{avg['x_rmse']:.2f}, "
        f"y MAE/RMSE={avg['y_mae']:.2f}/{avg['y_rmse']:.2f}, "
        f"z MAE/RMSE={avg['z_mae']:.2f}/{avg['z_rmse']:.2f}, "
        f"avg MAE/RMSE={avg['avg_mae']:.2f}/{avg['avg_rmse']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Path to MARS-main. Defaults to .benchmark/MARS/MARS-main.")
    parser.add_argument("--split", default="test", choices=("train", "validate", "test"))
    parser.add_argument("--prediction", default=None, help="Optional .npy prediction file with shape [N,57] or [N,19,3].")
    parser.add_argument("--json_out", default=None, help="Optional output JSON path for metric rows.")
    args = parser.parse_args()

    print("MARS dataset summary:")
    for row in describe(args.root):
        print(row)

    train = load_split(args.root, "train", mmap=True)
    target = load_split(args.root, args.split, mmap=True)
    if args.prediction:
        pred = _load_prediction(Path(args.prediction))
        source = args.prediction
    else:
        mean_pose = np.asarray(train.pose).mean(axis=0, keepdims=True)
        pred = np.repeat(mean_pose, target.pose.shape[0], axis=0)
        source = "train mean-pose baseline"

    if pred.shape != target.pose.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match target pose shape {target.pose.shape}")

    print(f"\nEvaluation split: {args.split}")
    print(f"Prediction source: {source}")
    print(f"MPJPE: {mpjpe(pred, target.pose) * 1000.0:.2f} mm")
    rows = mars_localization_table(pred, target.pose, MARS_JOINT_NAMES, scale=100.0)
    _print_mars_summary(rows)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump({"split": args.split, "prediction": source, "rows": rows}, handle, indent=2, ensure_ascii=False)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
