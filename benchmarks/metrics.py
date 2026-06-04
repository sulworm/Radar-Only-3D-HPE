"""Common pose metrics used by the public benchmark adapters."""

from __future__ import annotations

import numpy as np


def as_pose_array(values, name: str = "pose") -> np.ndarray:
    """Return a float array with shape [..., joints, 3]."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [..., joints, 3], got {array.shape}")
    return array


def joint_distances(pred, gt) -> np.ndarray:
    pred = as_pose_array(pred, "pred")
    gt = as_pose_array(gt, "gt")
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shapes differ: {pred.shape} != {gt.shape}")
    return np.linalg.norm(pred - gt, axis=-1)


def mpjpe(pred, gt) -> float:
    """Mean per-joint position error in the input coordinate unit."""
    return float(np.mean(joint_distances(pred, gt)))


def per_joint_mpjpe(pred, gt) -> np.ndarray:
    """Mean per-joint position error, preserving the joint axis."""
    distances = joint_distances(pred, gt)
    reduce_axes = tuple(range(distances.ndim - 1))
    return np.mean(distances, axis=reduce_axes)


def root_relative(pose, root_index: int = 0) -> np.ndarray:
    pose = as_pose_array(pose)
    return pose - pose[..., root_index : root_index + 1, :]


def root_relative_mpjpe(pred, gt, root_index: int = 0) -> float:
    return mpjpe(root_relative(pred, root_index), root_relative(gt, root_index))


def _similarity_transform(pred_frame: np.ndarray, gt_frame: np.ndarray, scale: bool = True) -> np.ndarray:
    """Align one predicted pose to ground truth with a Procrustes transform."""
    pred_frame = as_pose_array(pred_frame, "pred_frame")
    gt_frame = as_pose_array(gt_frame, "gt_frame")
    if pred_frame.shape != gt_frame.shape:
        raise ValueError(f"pred_frame and gt_frame shapes differ: {pred_frame.shape} != {gt_frame.shape}")

    mu_gt = gt_frame.mean(axis=0)
    mu_pred = pred_frame.mean(axis=0)
    gt_centered = gt_frame - mu_gt
    pred_centered = pred_frame - mu_pred

    norm_gt = np.linalg.norm(gt_centered)
    norm_pred = np.linalg.norm(pred_centered)
    if norm_gt < 1e-12 or norm_pred < 1e-12:
        return pred_frame + (mu_gt - mu_pred)

    gt_normalized = gt_centered / norm_gt
    pred_normalized = pred_centered / norm_pred
    matrix = gt_normalized.T @ pred_normalized
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    v = vt.T
    rotation = v @ u.T
    if np.linalg.det(rotation) < 0:
        v[:, -1] *= -1
        singular_values[-1] *= -1
        rotation = v @ u.T

    if scale:
        pose_scale = singular_values.sum() * norm_gt / norm_pred
    else:
        pose_scale = 1.0
    return pose_scale * pred_frame @ rotation + (mu_gt - pose_scale * mu_pred @ rotation)


def pa_mpjpe(pred, gt, scale: bool = True) -> float:
    """Procrustes-aligned MPJPE in the input coordinate unit."""
    pred = as_pose_array(pred, "pred")
    gt = as_pose_array(gt, "gt")
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shapes differ: {pred.shape} != {gt.shape}")

    pred_flat = pred.reshape(-1, pred.shape[-2], 3)
    gt_flat = gt.reshape(-1, gt.shape[-2], 3)
    aligned = np.empty_like(pred_flat, dtype=np.float64)
    for idx, (pred_frame, gt_frame) in enumerate(zip(pred_flat, gt_flat)):
        aligned[idx] = _similarity_transform(pred_frame, gt_frame, scale=scale)
    return mpjpe(aligned.reshape(pred.shape), gt)


def pck(pred, gt, thresholds=(0.05, 0.10, 0.15)) -> dict[str, float]:
    """Percentage of correct keypoints for thresholds in the input coordinate unit."""
    distances = joint_distances(pred, gt)
    return {f"pck@{threshold:g}": float(np.mean(distances <= threshold)) for threshold in thresholds}


def axis_mae_rmse(pred, gt) -> dict[str, np.ndarray | float]:
    """Axis-wise and joint-wise absolute/squared error summary."""
    pred = as_pose_array(pred, "pred")
    gt = as_pose_array(gt, "gt")
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shapes differ: {pred.shape} != {gt.shape}")
    diff = pred - gt
    abs_diff = np.abs(diff)
    sq_diff = diff**2
    reduce_axes = tuple(range(diff.ndim - 2))
    return {
        "axis_mae": np.mean(abs_diff, axis=reduce_axes + (-2,)),
        "axis_rmse": np.sqrt(np.mean(sq_diff, axis=reduce_axes + (-2,))),
        "joint_axis_mae": np.mean(abs_diff, axis=reduce_axes),
        "joint_axis_rmse": np.sqrt(np.mean(sq_diff, axis=reduce_axes)),
        "mean_axis_mae": float(np.mean(abs_diff)),
        "mean_axis_rmse": float(np.sqrt(np.mean(sq_diff))),
    }


def mars_localization_table(pred, gt, joint_names, scale: float = 100.0) -> list[dict[str, float | str]]:
    """Return MARS-style x/y/z MAE-RMSE rows, scaled to centimeters by default."""
    summary = axis_mae_rmse(pred, gt)
    joint_mae = np.asarray(summary["joint_axis_mae"]) * scale
    joint_rmse = np.asarray(summary["joint_axis_rmse"]) * scale
    if len(joint_names) != joint_mae.shape[0]:
        raise ValueError(f"Expected {joint_mae.shape[0]} joint names, got {len(joint_names)}")

    rows: list[dict[str, float | str]] = []
    for idx, name in enumerate(joint_names):
        rows.append(
            {
                "joint": name,
                "x_mae": float(joint_mae[idx, 0]),
                "x_rmse": float(joint_rmse[idx, 0]),
                "y_mae": float(joint_mae[idx, 1]),
                "y_rmse": float(joint_rmse[idx, 1]),
                "z_mae": float(joint_mae[idx, 2]),
                "z_rmse": float(joint_rmse[idx, 2]),
                "avg_mae": float(np.mean(joint_mae[idx])),
                "avg_rmse": float(np.mean(joint_rmse[idx])),
            }
        )

    axis_mae = np.asarray(summary["axis_mae"]) * scale
    axis_rmse = np.asarray(summary["axis_rmse"]) * scale
    rows.append(
        {
            "joint": "Average",
            "x_mae": float(axis_mae[0]),
            "x_rmse": float(axis_rmse[0]),
            "y_mae": float(axis_mae[1]),
            "y_rmse": float(axis_rmse[1]),
            "z_mae": float(axis_mae[2]),
            "z_rmse": float(axis_rmse[2]),
            "avg_mae": float(np.mean(axis_mae)),
            "avg_rmse": float(np.mean(axis_rmse)),
        }
    )
    return rows

