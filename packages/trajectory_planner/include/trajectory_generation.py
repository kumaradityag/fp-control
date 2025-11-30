import numpy as np
from typing import List, Tuple


def compute_centerline(
    yellow_pts: np.ndarray,
    white_pts: np.ndarray,
    max_forward: float,
    n_samples: int,
    lane_width: float,
) -> List[Tuple[float, float]]:
    if len(yellow_pts) < 2 or len(white_pts) < 2:
        return []

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

def polynomial_fit(points: np.ndarray, deg: int = 2, threshold: float = 0.5) -> np.ndarray:
    """
    Fits points to polynomial curve and remove outliers
    """
    # remove outlier points
    xs, ys = points[:, 0], points[:, 1]
    coeffs = np.polyfit(xs, ys, deg)
    fit_ys = np.polyval(coeffs, xs)
    mask = np.abs(ys - fit_ys) < threshold # FIXME: use std instead of threshold?
    points = points[mask]

    # polynomial fit
    xs, ys = points[:, 0], points[:, 1]
    coeffs = np.polyfit(xs, ys, deg)
    fit_ys = np.polyval(coeffs, xs)
    polynomial_points = np.column_stack((xs, fit_ys))

    # TODO: fit polynomial curve further

    return polynomial_points
