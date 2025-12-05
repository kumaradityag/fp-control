import numpy as np
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Optional


def get_spline_path(
    pts: np.ndarray, n_samples: int, smoothing: float = 0.0, max_forward: float = 0.6
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fits a B-spline to points and returns (x, y) coordinates and (dx, dy) derivatives.
    Returns None if fitting fails or insufficient points.
    """
    # Sort by distance to origin to ensure sequential ordering
    dists = np.linalg.norm(pts, axis=1)
    sorted_idx = np.argsort(dists)
    sorted_pts = pts[sorted_idx]

    # Filter duplicates to prevent spline errors (min 1cm dist)
    if len(sorted_pts) > 1:
        diffs = np.diff(sorted_pts, axis=0)
        dist_sq = np.sum(diffs**2, axis=1)
        keep_mask = np.concatenate(([True], dist_sq > 0.01**2))
        unique_pts = sorted_pts[keep_mask]
    else:
        unique_pts = sorted_pts

    count = len(unique_pts)
    if count < 2:
        return None

    try:
        k_degree = min(3, count - 1)
        # Use generous smoothing to handle noise
        s_val = count * 0.5 if smoothing == 0.0 else smoothing

        # Fit spline
        tck, u = splprep(unique_pts.T, s=s_val, k=k_degree)

        # Generate even parameter steps
        u_new = np.linspace(0, 1, n_samples)

        # Evaluate positions (0th derivative)
        x_new, y_new = splev(u_new, tck)
        # Evaluate tangents (1st derivative) for normal calculation
        dx_new, dy_new = splev(u_new, tck, der=1)

        return np.array(x_new), np.array(y_new), np.array(dx_new), np.array(dy_new)

    except Exception:
        return None


def compute_centerline(
    yellow_pts: np.ndarray,
    white_pts: np.ndarray,
    yellow_normals: np.ndarray,
    white_normals: np.ndarray,
    max_forward: float,
    n_samples: int,
    lane_width: float,
    epsilon: float,
) -> List[Tuple[float, float]]:
    """
    Computes centerline by fitting a spline ONLY to the yellow line and offsetting.
    White points are ignored as requested.
    """
    half_width = (lane_width / 2.0) + epsilon

    # --- 1. Filter Yellow Points (Range Only) ---
    # We remove the side check (y > 0) to allow the line to bend/cross anywhere.
    valid_y = np.array([])
    if yellow_pts.size > 0:
        mask = (
            (yellow_pts[:, 0] > 0)  # X forward
            & (yellow_pts[:, 0] < max_forward)  # X max forward
            & (yellow_pts[:, 1] > -half_width)  # Y ignore too far right
        )
        if np.any(mask):
            valid_y = yellow_pts[mask]

    # --- 2. Fit Spline to Yellow Line ---
    y_res = get_spline_path(valid_y, n_samples, max_forward)

    center_x, center_y = [], []

    # --- 3. Compute Centerline via Offset ---
    if y_res is not None:
        # Yellow line exists -> Shift Right using Spline Tangents
        # Tangent vector T = (dx, dy)
        # Normal vector N (Right/Clockwise) = (dy, -dx)
        yx, yy, ydx, ydy = y_res

        # Normalize tangents
        norms = np.hypot(ydx, ydy)
        norms[norms == 0] = 1.0  # Avoid division by zero

        # Unit Tangent Components
        ux = ydx / norms
        uy = ydy / norms

        # Shift Right (Clockwise rotation)
        # New X = Old X + (Normal X * offset) -> Normal X is uy
        # New Y = Old Y + (Normal Y * offset) -> Normal Y is -ux
        center_x = yx + (uy * half_width)
        center_y = yy + (-ux * half_width)

    else:
        # Case: No yellow line -> Project Straight Forward
        center_x = np.linspace(0, max_forward, n_samples)
        center_y = np.zeros(n_samples)

    return list(zip(center_x, center_y))
