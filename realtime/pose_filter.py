"""Optional pose smoothing for the real-time worker."""

import numpy as np

#0.008 and 0.05 are good defaults for smoothing the pose without introducing too much lag.
DEFAULT_PROCESS_NOISE = 0.008
DEFAULT_MEASUREMENT_NOISE = 0.05


class KalmanFilter1D:
    def __init__(
        self,
        process_noise=DEFAULT_PROCESS_NOISE,
        measurement_noise=DEFAULT_MEASUREMENT_NOISE,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.value = 0.0
        self.variance = 1.0
        self.initialized = False

    def update(self, measurement):
        measurement = float(measurement)
        if not self.initialized:
            self.value = measurement
            self.initialized = True
            return measurement

        predicted_variance = self.variance + self.process_noise
        gain = predicted_variance / (predicted_variance + self.measurement_noise)
        self.value += gain * (measurement - self.value)
        self.variance = (1.0 - gain) * predicted_variance
        return self.value


class MultiJointKalmanFilter:
    def __init__(
        self,
        num_joints=13,
        process_noise=DEFAULT_PROCESS_NOISE,
        measurement_noise=DEFAULT_MEASUREMENT_NOISE,
    ):
        self.filters = [
            [KalmanFilter1D(process_noise, measurement_noise) for _ in range(3)]
            for _ in range(num_joints)
        ]

    def smooth(self, pose):
        pose = np.asarray(pose, dtype=np.float32)
        smoothed = np.empty_like(pose)
        for joint_index, joint_filters in enumerate(self.filters):
            for axis, kalman_filter in enumerate(joint_filters):
                smoothed[joint_index, axis] = kalman_filter.update(pose[joint_index, axis])
        return smoothed
