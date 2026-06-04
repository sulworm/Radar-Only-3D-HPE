"""Minimal real-time radar pipeline without UI or model dependencies.

Flow:
    serial bytes -> packet parsing -> raw point cloud -> front-end correction
    -> local crop and sampling -> model-ready frame buffer

Run:
    python radar_pipeline.py --list-ports
    python radar_pipeline.py --port COM3
"""

import argparse
import math
import struct
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports


# Radar protocol settings copied from main6.1.py.
MAGIC = b"\x55\xAA"
RANGE_RES = 0.025
VELOCITY_RES = 0.1
POINT_TYPES = {1}
POINT_BYTES = 9
MAX_FRAME_BYTES = 1024 * 1024

# Front-end processing settings copied from main6.1.py.
DEFAULT_RADAR_HEIGHT = 1
DEFAULT_PITCH_DEG = 0
NUM_POINTS = 128
SEQ_LEN = 10
VELOCITY_CLIP = 1.5
PA_LOG_SCALE = 12.0
LOCAL_CROP_BOX = 1.5
GLOBAL_Z_RANGE = (-0.5, 2.5)
ROUGH_XY_RANGE = (-3.5, 3.5)
ROUGH_Z_RANGE = (-0.5, 2.5)

# Soft body tracking for sparse point clouds. Clustering locates the body core;
# it does not discard local context points before the model sees them.
BODY_CLUSTER_EPS = 0.45
BODY_CLUSTER_MIN_SAMPLES = 3
BODY_CLUSTER_LOOSE_EPS = 0.55
BODY_CLUSTER_LOOSE_MIN_SAMPLES = 2
BODY_CLUSTER_Z_SCALE = 0.5
BODY_ASSOCIATION_GATE = 0.8
BODY_HALO_RADIUS = 0.55
CENTER_EMA_ALPHA = 0.3
CENTER_MAX_STEP = 0.3
CENTER_REACQUIRE_AFTER = 3
SAMPLE_VOXEL_SIZE = 0.25

CATEGORY_CORE = 0
CATEGORY_HALO = 1
CATEGORY_CONTEXT = 2
CATEGORY_REPEAT_FACTORS = {
    CATEGORY_CORE: 1.00,
    CATEGORY_HALO: 1.15,
    CATEGORY_CONTEXT: 0.70,
}

NEED_MORE_DATA = 0
BAD_ALIGNMENT = 1
FRAME_READY = 2


def empty_points():
    return np.empty((0, 5), dtype=np.float32)


def parse_packet(data):
    """Parse one complete frame from the beginning of a byte buffer."""
    if len(data) < 10:
        return empty_points(), NEED_MORE_DATA, 0
    if not data.startswith(MAGIC):
        return empty_points(), BAD_ALIGNMENT, 0

    try:
        frame_len = struct.unpack_from("<I", data, 2)[0]
        total_len = 6 + frame_len
        if total_len < 9 or total_len > MAX_FRAME_BYTES:
            return empty_points(), BAD_ALIGNMENT, 0
        if len(data) < total_len:
            return empty_points(), NEED_MORE_DATA, 0

        frame = data[:total_len]
        offset = 9
        points = []

        while offset < total_len:
            if offset + 3 > total_len:
                return empty_points(), BAD_ALIGNMENT, 0

            point_type = frame[offset]
            point_count = struct.unpack_from("<H", frame, offset + 1)[0]
            offset += 3
            block_len = point_count * POINT_BYTES
            if offset + block_len > total_len:
                return empty_points(), BAD_ALIGNMENT, 0

            if point_type not in POINT_TYPES:
                offset += block_len
                continue

            for _ in range(point_count):
                distance_raw, velocity_raw, az_raw, el_raw, pa = struct.unpack_from(
                    "<HBBBI", frame, offset
                )
                offset += POINT_BYTES

                distance = distance_raw * RANGE_RES
                velocity = (velocity_raw - 32) * VELOCITY_RES
                azimuth = math.asin((az_raw if az_raw < 64 else az_raw - 128) / 64.0)
                elevation = math.asin((el_raw if el_raw < 64 else el_raw - 128) / 64.0)
                x = distance * math.cos(elevation) * math.cos(azimuth)
                y = distance * math.cos(elevation) * math.sin(azimuth)
                z = distance * math.sin(elevation)
                points.append([x, y, z, velocity, pa])

        return np.asarray(points, dtype=np.float32).reshape(-1, 5), FRAME_READY, total_len
    except (ValueError, struct.error):
        return empty_points(), BAD_ALIGNMENT, 0


def apply_frontend_correction(points, radar_height, pitch_deg):
    """Convert raw radar XYZ into world XYZ using pitch rotation and height shift."""
    corrected = points.copy()
    if corrected.shape[0] == 0:
        return corrected

    pitch = math.radians(pitch_deg)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    x = corrected[:, 0].copy()
    z = corrected[:, 2].copy()
    corrected[:, 0] = x * cos_pitch + z * sin_pitch
    corrected[:, 2] = -x * sin_pitch + z * cos_pitch + radar_height
    return corrected


def rough_point_mask(xyz):
    return (
        (xyz[:, 0] >= ROUGH_XY_RANGE[0])
        & (xyz[:, 0] <= ROUGH_XY_RANGE[1])
        & (xyz[:, 1] >= ROUGH_XY_RANGE[0])
        & (xyz[:, 1] <= ROUGH_XY_RANGE[1])
        & (xyz[:, 2] >= ROUGH_Z_RANGE[0])
        & (xyz[:, 2] <= ROUGH_Z_RANGE[1])
    )


def estimate_center(xyz, previous_center=None, ema_alpha=0.2):
    if xyz.shape[0] == 0:
        center = np.zeros(3, dtype=np.float32) if previous_center is None else previous_center.copy()
        return center

    valid_points = xyz[rough_point_mask(xyz)]
    if valid_points.shape[0] == 0:
        valid_points = xyz
    current_center = valid_points.mean(axis=0).astype(np.float32)
    current_center[2] = 0.0

    if previous_center is None:
        return current_center
    center = ema_alpha * current_center + (1.0 - ema_alpha) * previous_center
    center[2] = 0.0
    return center.astype(np.float32)


def select_radar_channels(points, input_channels=5):
    """Build xyzvpa features. The protocol meaning of pa still needs confirmation."""
    if input_channels not in (3, 4, 5):
        raise ValueError(f"Unsupported radar input channel count: {input_channels}")
    features = points[:, :input_channels].astype(np.float32, copy=True)
    if input_channels >= 4:
        features[:, 3] = np.clip(features[:, 3], -VELOCITY_CLIP, VELOCITY_CLIP) / VELOCITY_CLIP
    if input_channels >= 5:
        features[:, 4] = np.log1p(np.maximum(features[:, 4], 0.0)) / PA_LOG_SCALE
    return features


def crop_local_points(local_points):
    return local_points[local_point_mask(local_points)]


def local_point_mask(local_points):
    return (
        (local_points[:, 0] >= -LOCAL_CROP_BOX)
        & (local_points[:, 0] <= LOCAL_CROP_BOX)
        & (local_points[:, 1] >= -LOCAL_CROP_BOX)
        & (local_points[:, 1] <= LOCAL_CROP_BOX)
        & (local_points[:, 2] >= GLOBAL_Z_RANGE[0])
        & (local_points[:, 2] <= GLOBAL_Z_RANGE[1])
    )


def _limit_xy_step(delta, max_step=CENTER_MAX_STEP):
    norm = float(np.linalg.norm(delta))
    if norm <= max_step or norm <= 1e-9:
        return delta
    return delta / norm * max_step


def _anisotropic_distances(xyz_a, xyz_b):
    delta = xyz_a[:, None, :] - xyz_b[None, :, :]
    delta[:, :, 2] *= BODY_CLUSTER_Z_SCALE
    return np.linalg.norm(delta, axis=2)


def _dbscan_clusters(xyz, eps, min_samples):
    """Return DBSCAN-style index clusters for the small per-frame point cloud."""
    point_count = xyz.shape[0]
    if point_count < min_samples:
        return []

    adjacency = _anisotropic_distances(xyz, xyz) <= eps
    core_mask = adjacency.sum(axis=1) >= min_samples
    visited = np.zeros(point_count, dtype=bool)
    clusters = []

    for start in np.flatnonzero(core_mask):
        if visited[start]:
            continue
        stack = [int(start)]
        visited[start] = True
        core_indices = []
        while stack:
            index = stack.pop()
            core_indices.append(index)
            neighbors = np.flatnonzero(adjacency[index] & core_mask)
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        border_mask = adjacency[np.asarray(core_indices)].any(axis=0)
        clusters.append(np.flatnonzero(border_mask))
    return clusters


def _cluster_measurement(xyz, indices):
    points = xyz[indices]
    median = np.median(points, axis=0).astype(np.float32)
    median[2] = 0.0
    spread_xy = float(np.median(np.linalg.norm(points[:, :2] - median[:2], axis=1)))
    z_span = float(np.percentile(points[:, 2], 90) - np.percentile(points[:, 2], 10))
    quality = math.log1p(len(indices)) + 0.6 * min(z_span / 1.2, 1.0) - 0.35 * spread_xy
    return median, quality


def _select_body_cluster(xyz, previous_center, eps, min_samples):
    candidates = []
    for indices in _dbscan_clusters(xyz, eps, min_samples):
        measurement, quality = _cluster_measurement(xyz, indices)
        distance = 0.0
        if previous_center is not None:
            distance = float(np.linalg.norm(measurement[:2] - previous_center[:2]))
            quality -= 1.5 * min(distance, 2.0)
        candidates.append((quality, distance, indices, measurement))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0]


def _fallback_center(xyz, previous_center):
    if previous_center is not None:
        return previous_center.copy()
    if xyz.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    center = np.median(xyz, axis=0).astype(np.float32)
    center[2] = 0.0
    return center


def track_body_center(xyz, previous_center=None, missed_frames=0):
    """Track a robust XY body center and label core/halo/context points."""
    point_count = xyz.shape[0]
    categories = np.full(point_count, CATEGORY_CONTEXT, dtype=np.int8)
    if point_count == 0:
        return _fallback_center(xyz, previous_center), categories, missed_frames + 1, "empty"

    selected = _select_body_cluster(
        xyz,
        previous_center,
        BODY_CLUSTER_EPS,
        BODY_CLUSTER_MIN_SAMPLES,
    )
    if selected is None:
        selected = _select_body_cluster(
            xyz,
            previous_center,
            BODY_CLUSTER_LOOSE_EPS,
            BODY_CLUSTER_LOOSE_MIN_SAMPLES,
        )

    if selected is None:
        return _fallback_center(xyz, previous_center), categories, missed_frames + 1, "fallback"

    _, distance, core_indices, measurement = selected
    if (
        previous_center is not None
        and distance > BODY_ASSOCIATION_GATE
        and missed_frames < CENTER_REACQUIRE_AFTER
    ):
        return previous_center.copy(), categories, missed_frames + 1, "held"

    categories[core_indices] = CATEGORY_CORE
    non_core_indices = np.flatnonzero(categories != CATEGORY_CORE)
    if non_core_indices.size:
        distances = _anisotropic_distances(xyz[non_core_indices], xyz[core_indices])
        halo_indices = non_core_indices[np.min(distances, axis=1) <= BODY_HALO_RADIUS]
        categories[halo_indices] = CATEGORY_HALO

    if previous_center is None:
        center = measurement
    else:
        delta = measurement[:2] - previous_center[:2]
        delta = _limit_xy_step(delta)
        center = previous_center.copy()
        center[:2] += CENTER_EMA_ALPHA * delta
        center[2] = 0.0
    return center.astype(np.float32), categories, 0, "tracked"


def _voxel_keys(points):
    return np.floor(points[:, :3] / SAMPLE_VOXEL_SIZE).astype(np.int32)


def _point_repeat_weights(points, categories):
    """Build deterministic sparse-cloud repeat weights without hard Doppler cuts."""
    weights = np.zeros(points.shape[0], dtype=np.float64)
    voxel_keys = _voxel_keys(points)
    _, voxel_inverse, voxel_counts = np.unique(
        voxel_keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    spatial_weights = 1.0 / voxel_counts[voxel_inverse]
    motion_weights = np.ones(points.shape[0], dtype=np.float64)
    if points.shape[1] >= 4:
        motion_weights += 0.20 * np.abs(points[:, 3])
    if points.shape[1] >= 5:
        motion_weights += 0.10 * np.clip(points[:, 4], 0.0, 2.0)

    for category, factor in CATEGORY_REPEAT_FACTORS.items():
        weights[categories == category] = factor
    weights *= spatial_weights * motion_weights
    return weights


def _deterministic_repeat_indices(weights, repeat_count):
    if repeat_count <= 0:
        return np.empty(0, dtype=np.int64)
    expected = weights / weights.sum() * repeat_count
    counts = np.floor(expected).astype(np.int64)
    remainder = repeat_count - int(counts.sum())
    if remainder:
        fractions = expected - counts
        order = np.lexsort((np.arange(weights.size), -fractions))
        counts[order[:remainder]] += 1
    return np.repeat(np.arange(weights.size), counts)


def _spatially_balanced_indices(points, target_count):
    """Round-robin voxels so rare local regions survive if downsampling is needed."""
    voxel_keys = _voxel_keys(points)
    buckets = {}
    for index, key in enumerate(map(tuple, voxel_keys)):
        buckets.setdefault(key, []).append(index)

    selected = []
    depth = 0
    keys = sorted(buckets)
    while len(selected) < target_count:
        added = False
        for key in keys:
            bucket = buckets[key]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) == target_count:
                    break
        if not added:
            break
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def sample_or_pad(points, rng=None, categories=None):
    """Keep every sparse local point once, then fill deterministically to NUM_POINTS."""
    feature_count = points.shape[1] if points.ndim == 2 else 5
    point_count = points.shape[0]
    if point_count >= NUM_POINTS:
        indices = _spatially_balanced_indices(points, NUM_POINTS)
        return points[indices].astype(np.float32)
    if point_count > 0:
        if categories is None:
            categories = np.full(point_count, CATEGORY_CORE, dtype=np.int8)
        weights = _point_repeat_weights(points, categories)
        extra_indices = _deterministic_repeat_indices(weights, NUM_POINTS - point_count)
        extra = points[extra_indices]
        return np.vstack([points, extra]).astype(np.float32)
    return np.zeros((NUM_POINTS, feature_count), dtype=np.float32)


def sample_or_pad_legacy(points, rng):
    feature_count = points.shape[1] if points.ndim == 2 else 5
    point_count = points.shape[0]
    if point_count >= NUM_POINTS:
        return points[rng.choice(point_count, NUM_POINTS, replace=False)].astype(np.float32)
    if point_count > 0:
        extra = points[rng.choice(point_count, NUM_POINTS - point_count, replace=True)]
        return np.vstack([points, extra]).astype(np.float32)
    return np.zeros((NUM_POINTS, feature_count), dtype=np.float32)


@dataclass
class ProcessedFrame:
    raw_points: np.ndarray
    world_points: np.ndarray
    local_points: np.ndarray
    model_input: np.ndarray
    center: np.ndarray
    core_point_count: int = 0
    halo_point_count: int = 0
    context_point_count: int = 0
    tracking_status: str = "legacy"


class RadarFrameProcessor:
    def __init__(
        self,
        radar_height=DEFAULT_RADAR_HEIGHT,
        pitch_deg=DEFAULT_PITCH_DEG,
        seed=42,
        seq_len=SEQ_LEN,
        input_channels=5,
        use_soft_cluster=True,
    ):
        self.radar_height = radar_height
        self.pitch_deg = pitch_deg
        self.previous_center = None
        self.missed_center_frames = 0
        self.rng = np.random.default_rng(seed)
        self.seq_len = int(seq_len)
        self.input_channels = int(input_channels)
        self.use_soft_cluster = bool(use_soft_cluster)
        self.frame_buffer = deque(maxlen=self.seq_len)

    def process(self, raw_points):
        world_points = apply_frontend_correction(raw_points, self.radar_height, self.pitch_deg)
        xyz = world_points[:, :3]

        mask = rough_point_mask(xyz)
        if xyz.shape[0] > 0 and not np.any(mask):
            mask = np.ones(xyz.shape[0], dtype=bool)

        rough_points = world_points[mask]
        if self.use_soft_cluster:
            center, categories, self.missed_center_frames, tracking_status = track_body_center(
                rough_points[:, :3],
                self.previous_center,
                self.missed_center_frames,
            )
        else:
            center = estimate_center(xyz, self.previous_center)
            categories = np.full(rough_points.shape[0], CATEGORY_CONTEXT, dtype=np.int8)
            tracking_status = "legacy"
        self.previous_center = center

        local_points = select_radar_channels(rough_points, self.input_channels)
        local_points[:, :3] -= center
        crop_mask = local_point_mask(local_points)
        local_points = local_points[crop_mask]
        local_categories = categories[crop_mask]
        if self.use_soft_cluster:
            model_input = sample_or_pad(local_points, categories=local_categories)
        else:
            model_input = sample_or_pad_legacy(local_points, self.rng)
        self.frame_buffer.append(model_input)

        return ProcessedFrame(
            raw_points,
            world_points,
            local_points,
            model_input,
            center,
            core_point_count=int(np.sum(local_categories == CATEGORY_CORE)),
            halo_point_count=int(np.sum(local_categories == CATEGORY_HALO)),
            context_point_count=int(np.sum(local_categories == CATEGORY_CONTEXT)),
            tracking_status=tracking_status,
        )

    def model_window(self):
        """Return (seq_len, NUM_POINTS, 5) only after the temporal buffer is full."""
        if len(self.frame_buffer) < self.seq_len:
            return None
        return np.stack(self.frame_buffer, axis=0)


class RadarSerialReader:
    def __init__(self, port, baudrate=2_000_000, timeout=0.02):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.buffer = bytearray()

    def open(self):
        self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        print(f"[INFO] serial connected: {self.port}, baudrate={self.baudrate}")

    def start_radar(self):
        self.serial.write(b"AT+RESET\n")
        time.sleep(0.5)
        self.serial.write(b"AT+START\n")
        time.sleep(0.5)
        print("[INFO] radar start commands sent")

    def read_frame(self, should_stop=None):
        """Read one frame, or return None when an optional stop callback fires."""
        while True:
            if should_stop is not None and should_stop():
                return None

            if self.serial.in_waiting:
                self.buffer.extend(self.serial.read(self.serial.in_waiting))

                while True:
                    if should_stop is not None and should_stop():
                        return None

                    points, status, consumed_len = parse_packet(self.buffer)
                    if status == NEED_MORE_DATA:
                        break
                    if status == BAD_ALIGNMENT:
                        del self.buffer[0]
                        continue

                    del self.buffer[:consumed_len]
                    return points

            time.sleep(0.001)

    def close(self):
        if self.serial is not None and self.serial.is_open:
            try:
                self.serial.write(b"AT\n")
            except serial.SerialException:
                pass
            self.serial.close()
            print("[INFO] serial closed")


def list_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("[WARN] no serial ports found")
        return
    print("[INFO] available serial ports:")
    for port in ports:
        print(f"  {port.device:<8} {port.description}")


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal real-time mmWave radar reader")
    parser.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    parser.add_argument("--port", help="serial port, for example COM3")
    parser.add_argument("--baudrate", type=int, default=2_000_000)
    parser.add_argument("--radar-height", type=float, default=DEFAULT_RADAR_HEIGHT)
    parser.add_argument("--pitch-deg", type=float, default=DEFAULT_PITCH_DEG)
    parser.add_argument("--save-dir", type=Path, help="optional directory for raw .npy frames")
    parser.add_argument("--print-interval", type=float, default=0.5)
    parser.add_argument(
        "--legacy-frontend",
        action="store_true",
        help="use the previous mean-center and random-padding front-end",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_ports:
        list_ports()
        return
    if not args.port:
        list_ports()
        print("\nRun example: python radar_pipeline.py --port COM3")
        return

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    reader = RadarSerialReader(args.port, args.baudrate)
    processor = RadarFrameProcessor(
        args.radar_height,
        args.pitch_deg,
        use_soft_cluster=not args.legacy_frontend,
    )
    frame_count = 0
    started_at = time.perf_counter()
    last_print_at = started_at

    try:
        reader.open()
        reader.start_radar()
        while True:
            raw_points = reader.read_frame()
            frame = processor.process(raw_points)
            frame_count += 1

            if args.save_dir:
                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
                np.save(args.save_dir / f"{timestamp}.npy", raw_points)

            now = time.perf_counter()
            if now - last_print_at >= args.print_interval:
                fps = frame_count / (now - started_at)
                buffer_size = len(processor.frame_buffer)
                print(
                    f"[FRAME {frame_count:06d}] "
                    f"fps={fps:5.1f} raw={len(frame.raw_points):3d} "
                    f"local={len(frame.local_points):3d} "
                    f"body={frame.core_point_count:3d}+{frame.halo_point_count:3d} "
                    f"context={frame.context_point_count:3d} "
                    f"track={frame.tracking_status:<8} "
                    f"buffer={buffer_size:2d}/{SEQ_LEN} "
                    f"center=({frame.center[0]:+.2f}, {frame.center[1]:+.2f}, {frame.center[2]:+.2f})"
                )
                last_print_at = now
    except KeyboardInterrupt:
        print("\n[INFO] stopped by user")
    except serial.SerialException as exc:
        print(f"[ERROR] serial failure: {exc}")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
