import numpy as np
from collections import deque


class TrajectoryBuffer:
    def __init__(
        self, max_size=5, theta_threshold=0.15, smooth_alpha=0.5, max_reject_count=3
    ):
        """
        max_size: number of previous trajectories to store
        change_threshold: max allowed deviation (m) before rejecting current trajectory
        smooth_alpha: smoothing weight between prev and current trajectory
        max_reject_count: max consecutive rejections before accepting new trajectory anyway
        """
        self.max_size = max_size
        #  self.change_threshold = change_threshold
        self.theta_threshold = theta_threshold
        self.smooth_alpha = smooth_alpha
        self.max_reject_count = max_reject_count
        self.reject_count = 0

        self.traj_buffer = deque(maxlen=max_size)  # stores (xs, ys)
        self.coeff_buffer = deque(maxlen=max_size)  # stores polynomial coefficients

    def add(self, xs, ys, coeffs):
        """Push a new trajectory & coeffs into the buffer."""
        self.traj_buffer.append((xs, ys))
        self.coeff_buffer.append(coeffs)

    def last(self):
        """Returns the most recent stored trajectory or None."""
        if len(self.traj_buffer) == 0:
            return None
        return self.traj_buffer[-1]

    def last_coeffs(self):
        if len(self.coeff_buffer) == 0:
            return None
        return self.coeff_buffer[-1]

    def deviation(self, xs, ys):
        """
        Compute L2 deviation between current trajectory and last stored trajectory.
        Returns a single scalar.
        """
        prev = self.last()
        if prev is None:
            return 0.0

        prev_xs, prev_ys = prev
        sample_idx = len(prev_xs) / 2
        x_sample, y_sample = xs[sample_idx], ys[sample_idx]
        prev_x_sample, prev_y_sample = prev_xs[sample_idx], prev_ys[sample_idx]
        theta1 = np.arctan(y_sample / x_sample)
        theta2 = np.arctan(prev_y_sample / prev_x_sample)
        theta = np.rad2deg(theta2 - theta1)

        return theta

    def should_reject(self, xs, ys):
        """Return True if this trajectory is too different from previous."""

        dev = self.deviation(xs, ys)
        if dev > self.theta_threshold:
            self.reject_count += 1
        else:
            self.reject_count = 0

        if self.reject_count >= self.max_reject_count:
            self.reject_count = 0
            return False  # accept anyway

        return dev > self.theta_threshold

    def smooth(self, xs, ys):
        """
        Smooth current trajectory using previous trajectory.
        blended = alpha * current + (1 - alpha) * previous
        """
        dev = self.deviation(xs, ys)
        if dev > self.change_threshold:
            return xs, ys

        prev = self.last()
        if prev is None:
            return xs, ys

        prev_xs, prev_ys = prev
        alpha = self.smooth_alpha

        smooth_xs = alpha * xs + (1 - alpha) * prev_xs
        smooth_ys = alpha * ys + (1 - alpha) * prev_ys
        return smooth_xs, smooth_ys
