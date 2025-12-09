import numpy as np
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Optional


def ransac_polyfit(points, degree, threshold, iters):
    """
    Fit a polynomial to the given 2D points using RANSAC to be robust to outliers.
    threshold: max error to still consider inlier (meters)
    """
    if len(points) < degree + 2:
        return None  # not enough samples

    X = points[:, 0]
    Y = points[:, 1]
    N = len(points)

    best_inliers = []
    best_coeffs = None

    for _ in range(iters):
        # sample minimal set
        idx = np.random.choice(N, degree + 1, replace=False)
        x_s = X[idx]
        y_s = Y[idx]

        # fit poly to sample
        try:
            coeffs = np.polyfit(x_s, y_s, degree)
        except:
            continue

        # evaluate residuals
        y_pred = np.polyval(coeffs, X)
        residuals = np.abs(y_pred - Y)

        inliers = residuals < threshold

        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_coeffs = coeffs

    if best_coeffs is None:
        return None

    # final refit on all inliers
    final_coeffs = np.polyfit(X[best_inliers], Y[best_inliers], degree)
    return final_coeffs


def shift_poly_curve(xs, ys, dy_dx, offset):
    """
    Shift a curve y(x) by 'offset' meters along the curve normal.
    Positive offset => shift to robot's right.
    Negative offset => shift to robot's left.
    """
    # Unit tangent
    norms = np.sqrt(1.0 + dy_dx**2)
    ux = 1.0 / norms  # tangent x-component
    uy = dy_dx / norms  # tangent y-component

    # Right normal = (uy, -ux)
    nx = uy
    ny = -ux

    # Apply shift
    shifted_x = xs + nx * offset
    shifted_y = ys + ny * offset

    return shifted_x, shifted_y


def decide_and_smooth(xs, ys, coeffs, trajectory_buffer):
    """
    Uses global trajectory_buffer to:
        - fallback to prev if current is None
        - reject drastic trajectory changes
        - smooth with previous trajectory
        - update buffer
    """

    # Case 0: current failed, use previous
    if xs is None or ys is None:
        prev = trajectory_buffer.last()
        return prev

    # Case 1: no previous trajectory -> accept directly
    prev = trajectory_buffer.last()
    if prev is None:
        trajectory_buffer.add(xs, ys, coeffs)
        return xs, ys

    # Case 2: check if trajectory is too different
    if trajectory_buffer.should_reject(xs, ys):
        print("Rejecting current trajectory (drastic change)")
        return prev  # fallback to previous

    # Case 3: smooth the trajectory
    # xs_s, ys_s = trajectory_buffer.smooth(xs, ys)

    # Add smoothed trajectory to the buffer
    trajectory_buffer.add(xs, ys, coeffs)

    return xs, ys


def compute_centerline(
    yellow_pts,
    white_pts,
    traj_buffer,
    min_forward,
    max_forward,
    n_samples,
    lane_width,
    epsilon,
    poly_degree,
    ransac_max_iterations,
    ransac_distance_threshold,
    yellow_pts_threshold,
    white_pts_threshold,
    default_mode,
):
    half_width = (lane_width / 2.0) + epsilon

    # filter yellow and white points based on y coordinate
    if len(yellow_pts) > 0:
        yellow_pts = yellow_pts[yellow_pts[:, 1] > (-half_width + epsilon)]
        yellow_pts = yellow_pts[yellow_pts[:, 0] > min_forward]  # only forward points
    if len(white_pts) > 0:
        white_pts = white_pts[white_pts[:, 1] < (half_width - epsilon)]
        white_pts = white_pts[white_pts[:, 0] > min_forward]  # only forward points

    # x_min is the min x value of yellow points
    x_min = np.min(yellow_pts[:, 0]) if len(yellow_pts) > 0 else 0.0
    xs = np.linspace(x_min, max_forward, n_samples)

    default_mode = default_mode.upper()
    if default_mode not in ["WHITE", "YELLOW"]:
        default_mode = "WHITE"

    if default_mode == "WHITE" and len(white_pts) >= white_pts_threshold:
        mode = "WHITE"
    elif default_mode == "YELLOW" and len(yellow_pts) >= yellow_pts_threshold:
        mode = "YELLOW"
    elif len(white_pts) >= white_pts_threshold:
        mode = "WHITE"
    elif len(yellow_pts) >= yellow_pts_threshold:
        mode = "YELLOW"
    else:
        mode = "NONE"
        print("No sufficient points detected, empty centerline.")

    if mode == "WHITE":
        white_coeffs = ransac_polyfit(
            white_pts,
            degree=poly_degree,
            threshold=ransac_distance_threshold,
            iters=ransac_max_iterations,
        )
        y_white = np.polyval(white_coeffs, xs)
        d_white = np.polyval(np.polyder(white_coeffs), xs)
        cx, cy = shift_poly_curve(xs, y_white, d_white, -half_width)
        smoothed = decide_and_smooth(cx, cy, white_coeffs, traj_buffer)
        cx_s, cy_s = smoothed
        return list(zip(cx_s, cy_s))
    elif mode == "YELLOW":
        yellow_coeffs = ransac_polyfit(
            yellow_pts,
            degree=poly_degree,
            threshold=ransac_distance_threshold,
            iters=ransac_max_iterations,
        )
        y_yellow = np.polyval(yellow_coeffs, xs)
        d_yellow = np.polyval(np.polyder(yellow_coeffs), xs)
        cx, cy = shift_poly_curve(xs, y_yellow, d_yellow, +half_width)
        smoothed = decide_and_smooth(cx, cy, yellow_coeffs, traj_buffer)
        cx_s, cy_s = smoothed
        return list(zip(cx_s, cy_s))

    return list(zip(xs, np.zeros_like(xs)))
