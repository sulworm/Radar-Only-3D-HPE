import argparse
import math
import os

import numpy as np
from tqdm import tqdm


def _normalize_degrees(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def _rotation_matrix_x(degrees):
    radians = math.radians(float(degrees))
    c = math.cos(radians)
    s = math.sin(radians)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_y(degrees):
    radians = math.radians(float(degrees))
    c = math.cos(radians)
    s = math.sin(radians)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_z(degrees):
    radians = math.radians(float(degrees))
    c = math.cos(radians)
    s = math.sin(radians)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def euler_rotation_matrix(rot_x=0.0, rot_y=0.0, rot_z=0.0):
    # Apply X pitch first, then Y roll, then Z yaw.
    return _rotation_matrix_z(rot_z) @ _rotation_matrix_y(rot_y) @ _rotation_matrix_x(rot_x)


def parse_group_spec(group_spec):
    if not group_spec:
        return None

    group_ids = []
    for chunk in str(group_spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        if "-" in chunk:
            start_text, end_text = [part.strip() for part in chunk.split("-", 1)]
            if start_text.isdigit() and end_text.isdigit():
                start = int(start_text)
                end = int(end_text)
                step = 1 if start <= end else -1
                group_ids.extend(str(group_id) for group_id in range(start, end + step, step))
                continue

        group_ids.append(chunk)

    return group_ids


class DataCenterAlignerSmoothNoZ:
    IMU_INDICES = [0, 1, 2, 3, 5, 6, 7, 13, 14, 16, 18, 20, 22]
    DEFAULT_RADAR_ROTATION_GROUPS = "5-44"
    DEFAULT_IMU_ROTATION_GROUPS = "158-226"
    # Alignment flow:
    # radar clusters -> robust center candidates -> temporal center tracker
    # -> smoothed XY offset -> translate IMU skeleton toward the radar track.
    # IMU is never used as the absolute reference; it is only a weak hint when
    # multiple radar candidates have similar quality.

    def __init__(
        self,
        source_dir=None,
        target_dir=None,
        imu_rotation_groups=None,
        radar_rotation_mode="none",
        radar_rotation_groups=None,
        fixed_rotation=(0.0, 0.0, 0.0),
        rotation_pivot="auto",
        rotation_translation="auto",
        auto_yaw=True,
        rotation_sample_count=180,
        pitch_min=-89.0,
        pitch_max=89.0,
        pitch_step=1.0,
        yaw_min_gain=0.20,
        yaw_min_spread=0.08,
        centroid_filter_radius=1.5,
        radar_z_min=-0.35,
        radar_z_max=2.6,
        radar_track_radius=1.25,
        radar_min_points=5,
        radar_cluster_radius=0.75,
        radar_cluster_z_radius=1.25,
        radar_max_candidates=5,
        center_max_measurement_jump=0.9,
        center_max_step=0.35,
        center_max_accel=0.18,
        center_update_alpha=0.45,
        center_reacquire_alpha=0.90,
        center_reacquire_after=3,
        center_velocity_decay=0.85,
        center_smooth_window=9,
        center_transition_weight=0.65,
        center_reacquire_transition_scale=0.20,
        center_imu_weight=0.08,
        smooth_window=12,
        offset_max_step=0.5,
        save_radar="auto",
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.source_dir = source_dir if source_dir else os.path.join(base_dir, "Data_Matched")
        self.target_dir = target_dir if target_dir else os.path.join(base_dir, "Data_Aligned")

        if imu_rotation_groups is None:
            imu_rotation_groups = self.DEFAULT_IMU_ROTATION_GROUPS
        self.imu_rotation_groups = set(parse_group_spec(imu_rotation_groups) or [])
        self.radar_rotation_mode = radar_rotation_mode
        self.radar_rotation_groups = set(
            parse_group_spec(radar_rotation_groups or self.DEFAULT_RADAR_ROTATION_GROUPS) or []
        )
        self.fixed_rotation = tuple(float(value) for value in fixed_rotation)
        self.rotation_pivot = str(rotation_pivot)
        self.rotation_translation = str(rotation_translation)
        self.auto_yaw = bool(auto_yaw)
        self.rotation_sample_count = max(5, int(rotation_sample_count))
        self.pitch_min = float(pitch_min)
        self.pitch_max = float(pitch_max)
        self.pitch_step = max(0.1, float(pitch_step))
        self.yaw_min_gain = float(yaw_min_gain)
        self.yaw_min_spread = float(yaw_min_spread)
        self.centroid_filter_radius = float(centroid_filter_radius)
        self.radar_z_min = float(radar_z_min)
        self.radar_z_max = float(radar_z_max)
        self.radar_track_radius = float(radar_track_radius)
        self.radar_min_points = max(1, int(radar_min_points))
        self.radar_cluster_radius = max(0.05, float(radar_cluster_radius))
        self.radar_cluster_z_radius = max(0.05, float(radar_cluster_z_radius))
        self.radar_max_candidates = max(1, int(radar_max_candidates))
        self.center_max_measurement_jump = max(0.0, float(center_max_measurement_jump))
        self.center_max_step = max(0.0, float(center_max_step))
        self.center_max_accel = max(0.0, float(center_max_accel))
        self.center_update_alpha = min(1.0, max(0.0, float(center_update_alpha)))
        self.center_reacquire_alpha = min(1.0, max(0.0, float(center_reacquire_alpha)))
        self.center_reacquire_after = max(1, int(center_reacquire_after))
        self.center_velocity_decay = min(1.0, max(0.0, float(center_velocity_decay)))
        self.center_smooth_window = max(1, int(center_smooth_window))
        self.center_transition_weight = max(0.0, float(center_transition_weight))
        self.center_reacquire_transition_scale = min(
            1.0,
            max(0.0, float(center_reacquire_transition_scale)),
        )
        self.center_imu_weight = max(0.0, float(center_imu_weight))
        self.smooth_window = max(1, int(smooth_window))
        self.offset_max_step = max(0.0, float(offset_max_step))
        self.save_radar = str(save_radar)

        self.available_groups = []
        self._find_available_groups()

    def _sort_group_ids(self, group_ids):
        try:
            return sorted(group_ids, key=lambda x: int(x) if str(x).isdigit() else str(x))
        except TypeError:
            return sorted(group_ids)

    def select_groups(self, group_ids):
        wanted = {str(group_id) for group_id in group_ids}
        available = set(self.available_groups)
        missing = self._sort_group_ids(wanted - available)
        if missing:
            print(f"Warning: requested groups not found and will be skipped: {missing}")
        self.available_groups = [group for group in self.available_groups if group in wanted]

    def prepare_imu_core(self, imu_raw, group_id, frame_name):
        if imu_raw.ndim != 2 or imu_raw.shape[1] < 3:
            raise ValueError(f"Invalid IMU shape in group {group_id}, frame {frame_name}: {imu_raw.shape}")

        if imu_raw.shape[0] == len(self.IMU_INDICES):
            return imu_raw[:, :3].copy()

        if imu_raw.shape[0] > max(self.IMU_INDICES):
            return imu_raw[self.IMU_INDICES, :3].copy()

        raise ValueError(
            f"IMU frame has {imu_raw.shape[0]} joints, expected 13 pre-cut joints or at least "
            f"{max(self.IMU_INDICES) + 1} raw joints: group {group_id}, frame {frame_name}"
        )

    def _find_available_groups(self):
        if not os.path.exists(self.source_dir):
            print(f"Error: source directory does not exist: {self.source_dir}")
            return

        for item in os.listdir(self.source_dir):
            group_path = os.path.join(self.source_dir, item)
            imu_dir = os.path.join(group_path, "IMU")
            radar_dir = os.path.join(group_path, "Radar")
            if os.path.isdir(group_path) and os.path.isdir(imu_dir) and os.path.isdir(radar_dir):
                self.available_groups.append(item)
            elif os.path.isdir(group_path):
                print(f"Skip non-standard data group: {group_path}")

        self.available_groups = self._sort_group_ids(self.available_groups)
        print(f"Found {len(self.available_groups)} groups to process")

    def _needs_legacy_imu_rotation(self, group_id):
        return str(group_id) in self.imu_rotation_groups

    def _needs_radar_rotation(self, group_id):
        if self.radar_rotation_mode == "none":
            return False
        return str(group_id) in self.radar_rotation_groups

    def _resolve_rotation_pivot_mode(self):
        if self.rotation_pivot != "auto":
            return self.rotation_pivot
        if self.radar_rotation_mode == "fixed":
            return "frame_centroid"
        return "origin"

    def _wants_rotation_translation(self, pivot_mode):
        if self.rotation_translation == "none":
            return False
        if self.rotation_translation == "xy":
            return True
        return pivot_mode in {"origin", "group_centroid", "imu_group_centroid"}

    def _apply_legacy_imu_rotation(self, imu_raw):
        imu_rotated = imu_raw.copy()
        old_x = imu_rotated[:, 0].copy()
        old_y = imu_rotated[:, 1].copy()
        imu_rotated[:, 0] = -old_y
        imu_rotated[:, 1] = old_x
        return imu_rotated

    def get_centroid(self, points):
        if points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)
        return np.mean(points, axis=0)

    def get_robust_centroid(self, points):
        if points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        finite_mask = np.isfinite(points).all(axis=1)
        finite_points = points[finite_mask]
        if finite_points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        if finite_points.shape[0] < 8:
            return np.mean(finite_points, axis=0).astype(np.float32)

        median = np.median(finite_points, axis=0)
        distances = np.linalg.norm(finite_points[:, :2] - median[:2], axis=1)
        keep = distances <= np.percentile(distances, 85)
        if not np.any(keep):
            return np.mean(finite_points, axis=0).astype(np.float32)
        return np.mean(finite_points[keep], axis=0).astype(np.float32)

    def get_radar_alignment_centroid(self, radar_xyz, imu_centroid=None, previous_centroid=None):
        if radar_xyz.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        finite_mask = np.isfinite(radar_xyz).all(axis=1)
        finite_points = radar_xyz[finite_mask]
        if finite_points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        z_mask = (finite_points[:, 2] >= self.radar_z_min) & (finite_points[:, 2] <= self.radar_z_max)
        candidate_points = finite_points[z_mask]
        if candidate_points.shape[0] < self.radar_min_points:
            candidate_points = finite_points

        if previous_centroid is not None and candidate_points.shape[0] >= self.radar_min_points:
            distances = np.linalg.norm(candidate_points[:, :2] - previous_centroid[:2], axis=1)
            tracked = candidate_points[distances <= self.radar_track_radius]
            if tracked.shape[0] >= self.radar_min_points:
                return self.get_robust_centroid(tracked)

        if imu_centroid is not None and candidate_points.shape[0] >= self.radar_min_points:
            radius = self.centroid_filter_radius
            local_mask = (
                (np.abs(candidate_points[:, 0] - imu_centroid[0]) <= radius)
                & (np.abs(candidate_points[:, 1] - imu_centroid[1]) <= radius)
            )
            local_points = candidate_points[local_mask]
            if local_points.shape[0] >= self.radar_min_points:
                return self.get_robust_centroid(local_points)

        return self.get_robust_centroid(candidate_points)

    def measure_radar_person_center(self, radar_xyz):
        candidates = self.get_radar_center_candidates(radar_xyz)
        if not candidates:
            return np.zeros(3, dtype=np.float32), False, {"reason": "empty", "point_count": 0}

        best = candidates[0]
        return best["center"], True, {
            "reason": "ok",
            "point_count": best["point_count"],
            "quality": best["quality"],
            "candidate_count": len(candidates),
        }

    def _candidate_radar_points(self, radar_xyz):
        finite_mask = np.isfinite(radar_xyz).all(axis=1)
        finite_points = radar_xyz[finite_mask]
        if finite_points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), "nonfinite"

        z_mask = (finite_points[:, 2] >= self.radar_z_min) & (finite_points[:, 2] <= self.radar_z_max)
        candidate_points = finite_points[z_mask]
        if candidate_points.shape[0] < self.radar_min_points:
            candidate_points = finite_points
            return candidate_points.astype(np.float32), "z_fallback"
        return candidate_points.astype(np.float32), "ok"

    def _connected_components(self, points):
        count = points.shape[0]
        if count == 0:
            return []

        visited = np.zeros(count, dtype=bool)
        components = []
        for start_idx in range(count):
            if visited[start_idx]:
                continue

            queue = [start_idx]
            visited[start_idx] = True
            component = []
            while queue:
                point_idx = queue.pop()
                component.append(point_idx)
                delta_xy = points[:, :2] - points[point_idx, :2]
                xy_dist = np.linalg.norm(delta_xy, axis=1)
                z_dist = np.abs(points[:, 2] - points[point_idx, 2])
                neighbor_mask = (
                    (~visited)
                    & (xy_dist <= self.radar_cluster_radius)
                    & (z_dist <= self.radar_cluster_z_radius)
                )
                neighbors = np.flatnonzero(neighbor_mask)
                if neighbors.shape[0] > 0:
                    visited[neighbors] = True
                    queue.extend(neighbors.tolist())
            components.append(component)
        return components

    def _make_center_candidate(self, points, source):
        center = self.get_robust_centroid(points)
        point_count = int(points.shape[0])
        if point_count <= 1:
            spread_xy = 0.0
            z_span = 0.0
        else:
            spread_xy = float(np.median(np.linalg.norm(points[:, :2] - center[:2], axis=1)))
            z_span = float(np.percentile(points[:, 2], 90) - np.percentile(points[:, 2], 10))

        count_score = math.log1p(point_count)
        height_score = min(1.0, max(0.0, z_span / 1.2))
        compact_penalty = 0.20 * spread_xy
        z_penalty = 0.05 * abs(float(center[2]) - 1.2)
        low_center_penalty = 0.45 * max(0.0, 1.0 - float(center[2]))
        high_center_penalty = 0.20 * max(0.0, float(center[2]) - 2.25)
        source_penalty = 0.35 if source == "all_points" else 0.0
        quality = (
            count_score
            + 0.35 * height_score
            - compact_penalty
            - z_penalty
            - low_center_penalty
            - high_center_penalty
            - source_penalty
        )
        return {
            "center": center.astype(np.float32),
            "point_count": point_count,
            "spread_xy": spread_xy,
            "z_span": z_span,
            "quality": float(quality),
            "source": source,
        }

    def get_radar_center_candidates(self, radar_xyz):
        if radar_xyz.shape[0] == 0:
            return []

        points, source = self._candidate_radar_points(radar_xyz)
        if points.shape[0] < self.radar_min_points:
            return []

        components = self._connected_components(points)
        candidates = []
        for component in components:
            if len(component) < self.radar_min_points:
                continue
            component_points = points[np.asarray(component, dtype=int)]
            candidates.append(self._make_center_candidate(component_points, source="cluster"))

        if not candidates:
            candidates.append(self._make_center_candidate(points, source=source))

        candidates.sort(key=lambda item: item["quality"], reverse=True)
        return candidates[: self.radar_max_candidates]

    def _limit_xy_vector(self, vector, max_norm):
        norm = float(np.linalg.norm(vector))
        if max_norm <= 0.0 or norm <= max_norm:
            return vector
        return vector / (norm + 1e-9) * max_norm

    def _median_smooth_centers(self, centers):
        if centers.shape[0] < self.center_smooth_window or self.center_smooth_window <= 1:
            return centers

        half_win = self.center_smooth_window // 2
        smoothed = centers.copy()
        for frame_idx in range(centers.shape[0]):
            start = max(0, frame_idx - half_win)
            end = min(centers.shape[0], frame_idx + half_win + 1)
            smoothed[frame_idx, :2] = np.median(centers[start:end, :2], axis=0)
        return smoothed.astype(np.float32)

    def _limit_center_motion(self, centers):
        if centers.shape[0] <= 1:
            return centers.astype(np.float32)

        limited = centers.copy()
        previous_velocity = np.zeros(2, dtype=np.float32)
        for frame_idx in range(1, limited.shape[0]):
            delta = limited[frame_idx, :2] - limited[frame_idx - 1, :2]
            accel = delta - previous_velocity
            accel = self._limit_xy_vector(accel, self.center_max_accel)
            delta = previous_velocity + accel
            delta = self._limit_xy_vector(delta, self.center_max_step)
            limited[frame_idx, :2] = limited[frame_idx - 1, :2] + delta
            previous_velocity = delta.astype(np.float32)
        return limited.astype(np.float32)

    def _select_center_candidate(self, candidates, predicted, imu_center, missed_count):
        if not candidates:
            return None, False, None

        is_reacquire = missed_count >= self.center_reacquire_after
        transition_weight = self.center_transition_weight
        if is_reacquire:
            transition_weight *= self.center_reacquire_transition_scale

        scored = []
        for candidate in candidates:
            center = candidate["center"]
            distance_to_prediction = float(np.linalg.norm(center[:2] - predicted[:2]))
            distance_to_imu = float(np.linalg.norm(center[:2] - imu_center[:2]))
            score = candidate["quality"]
            score -= transition_weight * distance_to_prediction
            score -= self.center_imu_weight * min(distance_to_imu, 3.0)
            scored.append((score, distance_to_prediction, candidate))

        scored.sort(key=lambda item: item[0], reverse=True)
        _, best_distance, best_candidate = scored[0]
        if best_distance > self.center_max_measurement_jump and not is_reacquire:
            return None, False, best_candidate
        return best_candidate, is_reacquire, best_candidate

    def track_radar_centers(self, candidate_lists, imu_centers):
        frame_count = len(candidate_lists)
        if frame_count == 0:
            return np.zeros((0, 3), dtype=np.float32), {
                "invalid_frames": 0,
                "rejected_jumps": 0,
                "predicted_frames": 0,
                "reacquired_frames": 0,
            }

        valid_indices = [idx for idx, candidates in enumerate(candidate_lists) if candidates]
        if not valid_indices:
            return np.zeros((frame_count, 3), dtype=np.float32), {
                "invalid_frames": frame_count,
                "rejected_jumps": 0,
                "predicted_frames": frame_count,
                "reacquired_frames": 0,
            }

        centers = np.zeros((frame_count, 3), dtype=np.float32)
        first_valid = int(valid_indices[0])
        initial_candidates = candidate_lists[first_valid]
        initial_candidates = sorted(
            initial_candidates,
            key=lambda candidate: (
                candidate["quality"] - self.center_imu_weight * np.linalg.norm(candidate["center"][:2] - imu_centers[first_valid, :2])
            ),
            reverse=True,
        )
        current = initial_candidates[0]["center"].copy()
        velocity = np.zeros(2, dtype=np.float32)
        invalid_frames = 0
        rejected_jumps = 0
        predicted_frames = 0
        reacquired_frames = 0
        missed_count = 0

        for frame_idx in range(frame_count):
            predicted = current.copy()
            if missed_count >= self.center_reacquire_after:
                predicted[:2] = current[:2]
            else:
                predicted[:2] = current[:2] + velocity
            candidates = candidate_lists[frame_idx]
            selected, is_reacquire, best_candidate = self._select_center_candidate(
                candidates,
                predicted,
                imu_centers[frame_idx],
                missed_count,
            )

            if selected is not None:
                measurement = selected["center"]
                if is_reacquire:
                    velocity[:] = 0.0
                    predicted[:2] = current[:2]
                target = predicted.copy()
                update_alpha = self.center_reacquire_alpha if is_reacquire else self.center_update_alpha
                target[:2] = predicted[:2] + update_alpha * (measurement[:2] - predicted[:2])
                target[2] = measurement[2]
                missed_count = 0
                if is_reacquire:
                    reacquired_frames += 1
            else:
                if not candidates:
                    invalid_frames += 1
                else:
                    rejected_jumps += 1
                predicted_frames += 1
                missed_count += 1
                target = current.copy()
                if not candidates:
                    velocity = velocity * self.center_velocity_decay
                    target[:2] = current[:2] + velocity
                else:
                    velocity[:] = 0.0

            if frame_idx <= first_valid:
                if selected is not None:
                    current = selected["center"].copy()
                velocity[:] = 0.0
            else:
                delta = target[:2] - current[:2]
                accel = delta - velocity
                accel = self._limit_xy_vector(accel, self.center_max_accel)
                delta = velocity + accel
                delta = self._limit_xy_vector(delta, self.center_max_step)
                next_center = current.copy()
                next_center[:2] = current[:2] + delta
                next_center[2] = target[2]
                velocity = delta.astype(np.float32)
                current = next_center.astype(np.float32)

            centers[frame_idx] = current

        centers[:first_valid] = centers[first_valid]
        centers = self._median_smooth_centers(centers)
        centers = self._limit_center_motion(centers)
        return centers.astype(np.float32), {
            "invalid_frames": invalid_frames,
            "rejected_jumps": rejected_jumps,
            "predicted_frames": predicted_frames,
            "reacquired_frames": reacquired_frames,
        }

    def smooth_offsets(self, offsets):
        if len(offsets) < self.smooth_window or self.smooth_window <= 1:
            return offsets

        half_win = self.smooth_window // 2
        smoothed = np.zeros_like(offsets)
        for frame_idx in range(len(offsets)):
            start = max(0, frame_idx - half_win)
            end = min(len(offsets), frame_idx + half_win + 1)
            smoothed[frame_idx] = np.median(offsets[start:end], axis=0)

        if self.offset_max_step <= 0.0:
            return smoothed.astype(np.float32)

        limited = smoothed.copy()
        for frame_idx in range(1, len(limited)):
            delta_xy = limited[frame_idx, :2] - limited[frame_idx - 1, :2]
            step = float(np.linalg.norm(delta_xy))
            if step > self.offset_max_step:
                limited[frame_idx, :2] = (
                    limited[frame_idx - 1, :2] + delta_xy / (step + 1e-9) * self.offset_max_step
                )

        return limited.astype(np.float32)

    def _estimate_group_pivot(self, samples, pivot_mode):
        if pivot_mode == "group_centroid":
            centers = [self.get_robust_centroid(radar_xyz) for radar_xyz in samples["radar_frames"]]
            return np.median(np.asarray(centers, dtype=np.float32), axis=0).astype(np.float32)
        if pivot_mode == "imu_group_centroid":
            return np.median(samples["imu_centers_3d"], axis=0).astype(np.float32)
        return None

    def _get_sample_rotation_pivot(self, radar_xyz, imu_center_3d, pivot_mode, group_pivot=None):
        if pivot_mode == "origin":
            return None
        if pivot_mode == "frame_centroid":
            return self.get_robust_centroid(radar_xyz)
        if pivot_mode == "imu_frame_centroid":
            return imu_center_3d.astype(np.float32)
        if pivot_mode in {"group_centroid", "imu_group_centroid"}:
            return group_pivot
        return None

    def _get_frame_rotation_pivot(self, radar_xyz, imu_13, pivot_mode, group_pivot=None):
        imu_center_3d = self.get_centroid(imu_13[:, :3]).astype(np.float32)
        return self._get_sample_rotation_pivot(radar_xyz, imu_center_3d, pivot_mode, group_pivot)

    def _transform_xyz(self, radar_xyz, rotation_matrix, pivot=None):
        if pivot is None:
            return radar_xyz @ rotation_matrix.T
        return (radar_xyz - pivot) @ rotation_matrix.T + pivot

    def transform_radar(self, radar_raw, rotation_matrix, translation=None, pivot=None):
        if rotation_matrix is None or radar_raw.shape[0] == 0:
            return radar_raw

        transformed = radar_raw.copy()
        transformed_xyz = self._transform_xyz(radar_raw[:, :3].astype(np.float32), rotation_matrix, pivot)
        if translation is not None:
            transformed_xyz = transformed_xyz + translation
        transformed[:, :3] = transformed_xyz.astype(radar_raw.dtype, copy=False)
        return transformed

    def _sample_frame_indices(self, count):
        sample_count = min(count, self.rotation_sample_count)
        if sample_count <= 0:
            return np.array([], dtype=int)
        return np.linspace(0, count - 1, sample_count, dtype=int)

    def _load_rotation_samples(self, group_id, imu_dir, radar_dir, imu_files, radar_files, count):
        imu_centers = []
        imu_centers_3d = []
        radar_frames = []
        imu_z_values = []
        radar_z_values = []

        for frame_idx in self._sample_frame_indices(count):
            imu_raw = np.load(os.path.join(imu_dir, imu_files[frame_idx]))
            radar_raw = np.load(os.path.join(radar_dir, radar_files[frame_idx]))

            if self._needs_legacy_imu_rotation(group_id):
                imu_raw = self._apply_legacy_imu_rotation(imu_raw)

            try:
                imu_13 = self.prepare_imu_core(imu_raw, group_id, imu_files[frame_idx])
            except ValueError as exc:
                print(exc)
                return None

            radar_xyz = radar_raw[:, :3].astype(np.float32)
            if radar_xyz.shape[0] == 0:
                continue

            imu_centers.append(np.mean(imu_13[:, :2], axis=0))
            imu_centers_3d.append(np.mean(imu_13[:, :3], axis=0))
            radar_frames.append(radar_xyz)
            imu_z_values.extend(imu_13[:, 2].tolist())
            radar_z_values.extend(radar_xyz[:, 2].tolist())

        if not radar_frames or not imu_centers:
            return None

        return {
            "imu_centers": np.asarray(imu_centers, dtype=np.float32),
            "imu_centers_3d": np.asarray(imu_centers_3d, dtype=np.float32),
            "radar_frames": radar_frames,
            "imu_z": np.asarray(imu_z_values, dtype=np.float32),
            "radar_z": np.asarray(radar_z_values, dtype=np.float32),
        }

    def _score_pitch(self, samples, pitch_degrees, pivot_mode, group_pivot=None):
        pitch_matrix = _rotation_matrix_x(pitch_degrees)
        rotated_z = []
        for frame_idx, radar_xyz in enumerate(samples["radar_frames"]):
            pivot = self._get_sample_rotation_pivot(
                radar_xyz,
                samples["imu_centers_3d"][frame_idx],
                pivot_mode,
                group_pivot,
            )
            rotated = self._transform_xyz(radar_xyz, pitch_matrix, pivot)
            rotated_z.extend(rotated[:, 2].tolist())

        rotated_z = np.asarray(rotated_z, dtype=np.float32)
        imu_z = samples["imu_z"]
        if rotated_z.shape[0] < 5 or imu_z.shape[0] < 5:
            return np.inf, 0.0

        radar_quantiles = np.percentile(rotated_z, [10, 50, 90])
        imu_quantiles = np.percentile(imu_z, [10, 50, 90])

        radar_height = radar_quantiles[2] - radar_quantiles[0]
        imu_height = imu_quantiles[2] - imu_quantiles[0]
        valid_z_ratio = float(np.mean((rotated_z >= -0.5) & (rotated_z <= 2.5)))
        score = abs(radar_quantiles[1] - imu_quantiles[1])
        score += 0.35 * abs(radar_height - imu_height)
        if valid_z_ratio < 0.85:
            score += (0.85 - valid_z_ratio) * 2.0
        return float(score), valid_z_ratio

    def _estimate_pitch(self, samples, pivot_mode, group_pivot=None):
        best_pitch = 0.0
        best_score = np.inf
        best_valid_z = 0.0

        pitch = self.pitch_min
        while pitch <= self.pitch_max + 1e-6:
            score, valid_z_ratio = self._score_pitch(samples, pitch, pivot_mode, group_pivot)
            if score < best_score:
                best_score = score
                best_pitch = pitch
                best_valid_z = valid_z_ratio
            pitch += self.pitch_step

        return float(best_pitch), float(best_score), float(best_valid_z)

    def _fit_planar_yaw(self, radar_centers, imu_centers):
        finite = np.isfinite(radar_centers).all(axis=1) & np.isfinite(imu_centers).all(axis=1)
        radar_centers = radar_centers[finite]
        imu_centers = imu_centers[finite]
        if radar_centers.shape[0] < 5:
            return 0.0, {
                "enabled": False,
                "reason": "too_few_frames",
                "gain": 0.0,
                "error": np.inf,
                "base_error": np.inf,
                "radar_spread": 0.0,
                "imu_spread": 0.0,
            }

        radar_centered = radar_centers - np.mean(radar_centers, axis=0)
        imu_centered = imu_centers - np.mean(imu_centers, axis=0)
        radar_spread = float(np.sqrt(np.mean(np.sum(radar_centered**2, axis=1))))
        imu_spread = float(np.sqrt(np.mean(np.sum(imu_centered**2, axis=1))))

        covariance = radar_centered.T @ imu_centered
        try:
            u, _, vt = np.linalg.svd(covariance)
        except np.linalg.LinAlgError:
            return 0.0, {
                "enabled": False,
                "reason": "svd_failed",
                "gain": 0.0,
                "error": np.inf,
                "base_error": np.inf,
                "radar_spread": radar_spread,
                "imu_spread": imu_spread,
            }

        rotation_2d = vt.T @ u.T
        if np.linalg.det(rotation_2d) < 0:
            vt[-1, :] *= -1
            rotation_2d = vt.T @ u.T

        rotated = radar_centered @ rotation_2d.T + np.mean(imu_centers, axis=0)
        translated_only = radar_centers + (np.mean(imu_centers, axis=0) - np.mean(radar_centers, axis=0))
        error = float(np.median(np.linalg.norm(rotated - imu_centers, axis=1)))
        base_error = float(np.median(np.linalg.norm(translated_only - imu_centers, axis=1)))
        gain = (base_error - error) / (base_error + 1e-9)
        yaw = _normalize_degrees(math.degrees(math.atan2(rotation_2d[1, 0], rotation_2d[0, 0])))

        enabled = (
            self.auto_yaw
            and gain >= self.yaw_min_gain
            and radar_spread >= self.yaw_min_spread
            and imu_spread >= self.yaw_min_spread
        )
        reason = "ok" if enabled else "low_confidence"
        return (yaw if enabled else 0.0), {
            "enabled": enabled,
            "reason": reason,
            "gain": float(gain),
            "error": error,
            "base_error": base_error,
            "radar_spread": radar_spread,
            "imu_spread": imu_spread,
            "raw_yaw": yaw,
        }

    def _estimate_radar_translation(self, samples, rotation_matrix, pivot_mode, group_pivot=None):
        if not self._wants_rotation_translation(pivot_mode):
            return np.zeros(3, dtype=np.float32)

        radar_centers = []
        for frame_idx, radar_xyz in enumerate(samples["radar_frames"]):
            pivot = self._get_sample_rotation_pivot(
                radar_xyz,
                samples["imu_centers_3d"][frame_idx],
                pivot_mode,
                group_pivot,
            )
            rotated_xyz = self._transform_xyz(radar_xyz, rotation_matrix, pivot)
            radar_centers.append(self.get_robust_centroid(rotated_xyz)[:2])

        radar_centers = np.asarray(radar_centers, dtype=np.float32)
        imu_centers = samples["imu_centers"]
        finite = np.isfinite(radar_centers).all(axis=1) & np.isfinite(imu_centers).all(axis=1)
        if not np.any(finite):
            return np.zeros(3, dtype=np.float32)

        xy_translation = np.median(imu_centers[finite] - radar_centers[finite], axis=0)
        return np.array([xy_translation[0], xy_translation[1], 0.0], dtype=np.float32)

    def _estimate_auto_radar_rotation(self, group_id, imu_dir, radar_dir, imu_files, radar_files, count):
        samples = self._load_rotation_samples(group_id, imu_dir, radar_dir, imu_files, radar_files, count)
        if not samples:
            return None, None, {"mode": "auto", "enabled": False, "reason": "no_samples"}

        pivot_mode = self._resolve_rotation_pivot_mode()
        group_pivot = self._estimate_group_pivot(samples, pivot_mode)

        pitch, pitch_score, valid_z_ratio = self._estimate_pitch(samples, pivot_mode, group_pivot)
        pitch_matrix = _rotation_matrix_x(pitch)

        rotated_radar_centers = []
        for frame_idx, radar_xyz in enumerate(samples["radar_frames"]):
            pivot = self._get_sample_rotation_pivot(
                radar_xyz,
                samples["imu_centers_3d"][frame_idx],
                pivot_mode,
                group_pivot,
            )
            rotated_xyz = self._transform_xyz(radar_xyz, pitch_matrix, pivot)
            rotated_radar_centers.append(self.get_robust_centroid(rotated_xyz)[:2])
        rotated_radar_centers = np.asarray(rotated_radar_centers, dtype=np.float32)

        yaw, yaw_report = self._fit_planar_yaw(rotated_radar_centers, samples["imu_centers"])
        rotation_matrix = euler_rotation_matrix(rot_x=pitch, rot_y=0.0, rot_z=yaw)
        translation = self._estimate_radar_translation(samples, rotation_matrix, pivot_mode, group_pivot)
        report = {
            "mode": "auto",
            "enabled": True,
            "pitch": float(pitch),
            "yaw": float(yaw),
            "pivot_mode": pivot_mode,
            "group_pivot": group_pivot.tolist() if group_pivot is not None else None,
            "translation": translation.tolist(),
            "pitch_score": float(pitch_score),
            "valid_z_ratio": float(valid_z_ratio),
            "yaw_report": yaw_report,
        }
        return rotation_matrix, translation, report

    def _get_radar_rotation(self, group_id, imu_dir, radar_dir, imu_files, radar_files, count):
        if not self._needs_radar_rotation(group_id):
            return None, None, {"mode": self.radar_rotation_mode, "enabled": False, "reason": "group_not_selected"}

        if self.radar_rotation_mode == "fixed":
            rot_x, rot_y, rot_z = self.fixed_rotation
            rotation_matrix = euler_rotation_matrix(rot_x=rot_x, rot_y=rot_y, rot_z=rot_z)
            samples = self._load_rotation_samples(group_id, imu_dir, radar_dir, imu_files, radar_files, count)
            pivot_mode = self._resolve_rotation_pivot_mode()
            group_pivot = self._estimate_group_pivot(samples, pivot_mode) if samples else None
            translation = (
                self._estimate_radar_translation(samples, rotation_matrix, pivot_mode, group_pivot)
                if samples
                else np.zeros(3, dtype=np.float32)
            )
            return rotation_matrix, translation, {
                "mode": "fixed",
                "enabled": True,
                "rot_x": float(rot_x),
                "rot_y": float(rot_y),
                "rot_z": float(rot_z),
                "pivot_mode": pivot_mode,
                "group_pivot": group_pivot.tolist() if group_pivot is not None else None,
                "translation": translation.tolist(),
            }

        if self.radar_rotation_mode == "auto":
            return self._estimate_auto_radar_rotation(group_id, imu_dir, radar_dir, imu_files, radar_files, count)

        return None, None, {"mode": self.radar_rotation_mode, "enabled": False, "reason": "disabled"}

    def process_group(self, group_id):
        source_group = os.path.join(self.source_dir, str(group_id))
        target_group = os.path.join(self.target_dir, str(group_id))
        source_imu_dir = os.path.join(source_group, "IMU")
        source_radar_dir = os.path.join(source_group, "Radar")
        if not os.path.isdir(source_imu_dir) or not os.path.isdir(source_radar_dir):
            print(f"Skip group missing IMU/Radar: {source_group}")
            return

        os.makedirs(os.path.join(target_group, "IMU"), exist_ok=True)
        os.makedirs(os.path.join(target_group, "Radar"), exist_ok=True)

        imu_files = sorted([f for f in os.listdir(source_imu_dir) if f.endswith(".npy")])
        radar_files = sorted([f for f in os.listdir(source_radar_dir) if f.endswith(".npy")])
        count = min(len(imu_files), len(radar_files))
        if count == 0:
            print(f"Skip empty group: {source_group}")
            return

        radar_rotation, radar_translation, rotation_report = self._get_radar_rotation(
            group_id, source_imu_dir, source_radar_dir, imu_files, radar_files, count
        )
        if rotation_report.get("enabled"):
            if rotation_report["mode"] == "auto":
                yaw_info = rotation_report["yaw_report"]
                print(
                    f"Group {group_id}: radar auto rotation "
                    f"pitch={rotation_report['pitch']:.1f}, yaw={rotation_report['yaw']:.1f}, "
                    f"pivot={rotation_report['pivot_mode']}, "
                    f"shift=({rotation_report['translation'][0]:.2f}, {rotation_report['translation'][1]:.2f}), "
                    f"z_valid={rotation_report['valid_z_ratio']:.2f}, "
                    f"yaw_gain={yaw_info.get('gain', 0.0):.2f}, yaw={yaw_info.get('reason')}"
                )
            else:
                print(
                    f"Group {group_id}: radar fixed rotation "
                    f"x={rotation_report['rot_x']:.1f}, y={rotation_report['rot_y']:.1f}, "
                    f"z={rotation_report['rot_z']:.1f}, pivot={rotation_report['pivot_mode']}, "
                    f"shift=({rotation_report['translation'][0]:.2f}, {rotation_report['translation'][1]:.2f})"
                )

        need_legacy_imu_rotation = self._needs_legacy_imu_rotation(group_id)
        pivot_mode = rotation_report.get("pivot_mode", "origin")
        group_pivot = rotation_report.get("group_pivot")
        group_pivot = np.asarray(group_pivot, dtype=np.float32) if group_pivot is not None else None
        imu_data_cache = []
        radar_data_cache = []
        imu_centers = []
        radar_candidate_lists = []

        for frame_idx in range(count):
            imu_raw = np.load(os.path.join(source_imu_dir, imu_files[frame_idx]))
            radar_raw = np.load(os.path.join(source_radar_dir, radar_files[frame_idx]))

            if need_legacy_imu_rotation:
                imu_raw = self._apply_legacy_imu_rotation(imu_raw)

            try:
                imu_13 = self.prepare_imu_core(imu_raw, group_id, imu_files[frame_idx])
            except ValueError as exc:
                print(exc)
                return

            radar_pivot = None
            if radar_rotation is not None:
                radar_pivot = self._get_frame_rotation_pivot(
                    radar_raw[:, :3].astype(np.float32),
                    imu_13,
                    pivot_mode,
                    group_pivot,
                )
            radar_aligned = self.transform_radar(radar_raw, radar_rotation, radar_translation, radar_pivot)
            imu_data_cache.append(imu_13)
            radar_data_cache.append(radar_aligned)
            c_imu = self.get_centroid(imu_13[:, :3])
            imu_centers.append(c_imu)

            if radar_aligned.shape[0] > 0:
                radar_xyz = radar_aligned[:, :3]
                radar_candidate_lists.append(self.get_radar_center_candidates(radar_xyz))
            else:
                radar_candidate_lists.append([])

        imu_centers = np.asarray(imu_centers, dtype=np.float32)
        has_candidates = any(bool(candidates) for candidates in radar_candidate_lists)
        if has_candidates:
            radar_centers, tracker_report = self.track_radar_centers(radar_candidate_lists, imu_centers)
            raw_offsets = radar_centers - imu_centers
            raw_offsets[:, 2] = 0.0
            if (
                tracker_report["predicted_frames"]
                or tracker_report["rejected_jumps"]
                or tracker_report["reacquired_frames"]
            ):
                print(
                    f"Group {group_id}: center tracker predicted={tracker_report['predicted_frames']}, "
                    f"rejected_jumps={tracker_report['rejected_jumps']}, "
                    f"reacquired={tracker_report['reacquired_frames']}, "
                    f"invalid={tracker_report['invalid_frames']}/{count}"
                )
        else:
            raw_offsets = np.zeros_like(imu_centers, dtype=np.float32)
            print(f"Group {group_id}: no reliable radar centers; keeping IMU XY unchanged")

        smoothed_offsets = self.smooth_offsets(raw_offsets)
        same_radar_dir = os.path.abspath(source_radar_dir) == os.path.abspath(os.path.join(target_group, "Radar"))
        save_radar = self.save_radar == "always" or (
            self.save_radar == "auto" and (radar_rotation is not None or not same_radar_dir)
        )

        for frame_idx in range(count):
            offset = smoothed_offsets[frame_idx]
            imu_final = imu_data_cache[frame_idx].copy()
            imu_final[:, 0] += offset[0]
            imu_final[:, 1] += offset[1]

            np.save(os.path.join(target_group, "IMU", imu_files[frame_idx]), imu_final)
            if save_radar:
                np.save(os.path.join(target_group, "Radar", radar_files[frame_idx]), radar_data_cache[frame_idx])

    def run(self):
        print("=== Start data alignment ===")
        print(
            "Logic: read -> optional radar rotation -> configured IMU xy rotation -> "
            "13-joint trim -> XY centroid alignment, keep IMU Z -> smooth -> save"
        )
        if self.imu_rotation_groups:
            print(
                "IMU XY rotation groups="
                f"{self._sort_group_ids(self.imu_rotation_groups)} "
                "(x'=-y, y'=x; moves mocap axes into the radar XY basis)"
            )
        print(
            f"Radar centroid: z_range=({self.radar_z_min:.2f}, {self.radar_z_max:.2f}), "
            f"track_radius={self.radar_track_radius:.2f}, min_points={self.radar_min_points}, "
            f"cluster_radius={self.radar_cluster_radius:.2f}, max_candidates={self.radar_max_candidates}"
        )
        print(
            f"Center tracker: reject_jump>{self.center_max_measurement_jump:.2f}m, "
            f"max_step={self.center_max_step:.2f}m/frame, "
            f"max_accel={self.center_max_accel:.2f}m/frame^2, "
            f"alpha={self.center_update_alpha:.2f}, reacquire_alpha={self.center_reacquire_alpha:.2f}, "
            f"reacquire_after={self.center_reacquire_after}, decay={self.center_velocity_decay:.2f}, "
            f"smooth_window={self.center_smooth_window}, "
            f"reacquire_transition_scale={self.center_reacquire_transition_scale:.2f}"
        )
        print(
            f"Offset smoothing: median_window={self.smooth_window}, "
            f"max_xy_step={self.offset_max_step:.2f} m/frame"
        )
        if self.radar_rotation_mode != "none":
            print(
                f"Radar rotation mode={self.radar_rotation_mode}, "
                f"groups={self._sort_group_ids(self.radar_rotation_groups)}, "
                f"pivot={self.rotation_pivot}, translation={self.rotation_translation}"
            )

        for group in tqdm(self.available_groups):
            self.process_group(group)

        print(f"Done. Aligned data saved to: {self.target_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default=None)
    parser.add_argument("--target_dir", default=None)
    parser.add_argument("--groups", default=None, help="Group ids or ranges, for example: 59 or 5-44")
    parser.add_argument(
        "--refine_aligned",
        action="store_true",
        help=(
            "Refine an existing Data_Aligned directory in place. This disables IMU XY re-rotation "
            "unless --imu_rotation_groups is explicitly supplied."
        ),
    )
    parser.add_argument(
        "--imu_rotation_groups",
        default=DataCenterAlignerSmoothNoZ.DEFAULT_IMU_ROTATION_GROUPS,
        help="Groups whose IMU XY axes should be rotated into the radar basis using x'=-y, y'=x.",
    )
    parser.add_argument(
        "--radar_rotation",
        default="none",
        choices=["auto", "fixed", "none"],
        help="Radar point-cloud rotation correction mode. Auto only affects --rotation_groups.",
    )
    parser.add_argument(
        "--rotation_groups",
        default=DataCenterAlignerSmoothNoZ.DEFAULT_RADAR_ROTATION_GROUPS,
        help="Groups that should receive radar rotation correction.",
    )
    parser.add_argument("--rot_x", type=float, default=0.0, help="Fixed radar rotation around X in degrees.")
    parser.add_argument("--rot_y", type=float, default=0.0, help="Fixed radar rotation around Y in degrees.")
    parser.add_argument("--rot_z", type=float, default=0.0, help="Fixed radar rotation around Z in degrees.")
    parser.add_argument(
        "--rotation_pivot",
        default="auto",
        choices=[
            "auto",
            "origin",
            "frame_centroid",
            "group_centroid",
            "imu_frame_centroid",
            "imu_group_centroid",
        ],
        help="Pivot used for radar rotation. Auto uses frame_centroid for fixed mode and origin for auto mode.",
    )
    parser.add_argument(
        "--rotation_translation",
        default="auto",
        choices=["auto", "xy", "none"],
        help="Optional group-level XY translation after rotation. Auto skips it for frame pivots.",
    )
    parser.add_argument("--no_auto_yaw", action="store_true", help="Disable auto yaw after auto pitch estimation.")
    parser.add_argument("--rotation_samples", type=int, default=180, help="Max sampled frames for auto rotation.")
    parser.add_argument("--pitch_min", type=float, default=-89.0)
    parser.add_argument("--pitch_max", type=float, default=89.0)
    parser.add_argument("--pitch_step", type=float, default=1.0)
    parser.add_argument("--yaw_min_gain", type=float, default=0.20)
    parser.add_argument("--yaw_min_spread", type=float, default=0.08)
    parser.add_argument("--centroid_filter_radius", type=float, default=1.5)
    parser.add_argument("--radar_z_min", type=float, default=-0.35)
    parser.add_argument("--radar_z_max", type=float, default=2.6)
    parser.add_argument("--radar_track_radius", type=float, default=1.25)
    parser.add_argument("--radar_min_points", type=int, default=5)
    parser.add_argument("--radar_cluster_radius", type=float, default=0.75)
    parser.add_argument("--radar_cluster_z_radius", type=float, default=1.25)
    parser.add_argument("--radar_max_candidates", type=int, default=5)
    parser.add_argument("--center_max_measurement_jump", type=float, default=0.9)
    parser.add_argument("--center_max_step", type=float, default=0.35)
    parser.add_argument("--center_max_accel", type=float, default=0.18)
    parser.add_argument("--center_update_alpha", type=float, default=0.45)
    parser.add_argument("--center_reacquire_alpha", type=float, default=0.90)
    parser.add_argument("--center_reacquire_after", type=int, default=3)
    parser.add_argument("--center_velocity_decay", type=float, default=0.85)
    parser.add_argument("--center_smooth_window", type=int, default=9)
    parser.add_argument("--center_transition_weight", type=float, default=0.65)
    parser.add_argument("--center_reacquire_transition_scale", type=float, default=0.20)
    parser.add_argument("--center_imu_weight", type=float, default=0.08)
    parser.add_argument("--smooth_window", type=int, default=12)
    parser.add_argument("--offset_max_step", type=float, default=0.5)
    parser.add_argument(
        "--save_radar",
        default="auto",
        choices=["auto", "always", "never"],
        help="Save radar frames. Auto skips radar writes for in-place refinement without radar rotation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = args.source_dir
    target_dir = args.target_dir
    imu_rotation_groups = args.imu_rotation_groups

    if args.refine_aligned:
        if source_dir is None:
            source_dir = os.path.join(base_dir, "Data_Aligned")
        if target_dir is None:
            target_dir = source_dir
        if imu_rotation_groups == DataCenterAlignerSmoothNoZ.DEFAULT_IMU_ROTATION_GROUPS:
            imu_rotation_groups = ""

    aligner = DataCenterAlignerSmoothNoZ(
        source_dir=source_dir,
        target_dir=target_dir,
        imu_rotation_groups=imu_rotation_groups,
        radar_rotation_mode=args.radar_rotation,
        radar_rotation_groups=args.rotation_groups,
        fixed_rotation=(args.rot_x, args.rot_y, args.rot_z),
        rotation_pivot=args.rotation_pivot,
        rotation_translation=args.rotation_translation,
        auto_yaw=not args.no_auto_yaw,
        rotation_sample_count=args.rotation_samples,
        pitch_min=args.pitch_min,
        pitch_max=args.pitch_max,
        pitch_step=args.pitch_step,
        yaw_min_gain=args.yaw_min_gain,
        yaw_min_spread=args.yaw_min_spread,
        centroid_filter_radius=args.centroid_filter_radius,
        radar_z_min=args.radar_z_min,
        radar_z_max=args.radar_z_max,
        radar_track_radius=args.radar_track_radius,
        radar_min_points=args.radar_min_points,
        radar_cluster_radius=args.radar_cluster_radius,
        radar_cluster_z_radius=args.radar_cluster_z_radius,
        radar_max_candidates=args.radar_max_candidates,
        center_max_measurement_jump=args.center_max_measurement_jump,
        center_max_step=args.center_max_step,
        center_max_accel=args.center_max_accel,
        center_update_alpha=args.center_update_alpha,
        center_reacquire_alpha=args.center_reacquire_alpha,
        center_reacquire_after=args.center_reacquire_after,
        center_velocity_decay=args.center_velocity_decay,
        center_smooth_window=args.center_smooth_window,
        center_transition_weight=args.center_transition_weight,
        center_reacquire_transition_scale=args.center_reacquire_transition_scale,
        center_imu_weight=args.center_imu_weight,
        smooth_window=args.smooth_window,
        offset_max_step=args.offset_max_step,
        save_radar=args.save_radar,
    )
    selected_groups = parse_group_spec(args.groups)
    if selected_groups:
        aligner.select_groups(selected_groups)
    aligner.run()
