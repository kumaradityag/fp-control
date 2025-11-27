import numpy as np
from typing import List, Tuple


def compute_centerline(
    yellow_pts: np.ndarray,
    white_pts: np.ndarray,
    yellow_normals: np.ndarray,
    white_normals: np.ndarray,
    max_forward: float,
    n_samples: int,
    lane_width: float,
) -> List[Tuple[float, float]]:
    x_samples = np.linspace(0.05, max_forward, n_samples)

    centerline = []

    for x in x_samples:
        eps = 0.07
        y_y = yellow_pts[(yellow_pts[:, 0] > x - eps) & (yellow_pts[:, 0] < x + eps)]
        y_w = white_pts[(white_pts[:, 0] > x - eps) & (white_pts[:, 0] < x + eps)]
        if len(y_y) == 0 or len(y_w) == 0:
            continue

        y_center = 0.5 * (np.mean(y_y[:, 1]) + np.mean(y_w[:, 1]))
        centerline.append((x, y_center))

    return centerline
