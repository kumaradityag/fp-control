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
    max_forward,
    n_samples,
    lane_width,
    epsilon,
    poly_degree,
    ransac_max_iterations,
    ransac_distance_threshold,
    yellow_pts_threshold,
):
    half_width = (lane_width / 2.0) + epsilon

    # filter yellow and white points based on y coordinate
    yellow_pts = yellow_pts[yellow_pts[:, 1] > (-half_width + epsilon)]
    white_pts = white_pts[white_pts[:, 1] < (half_width - epsilon)]

    # x_min is the min x value of yellow points
    x_min = np.min(yellow_pts[:, 0]) if len(yellow_pts) > 0 else 0.0

    xs = np.linspace(x_min, max_forward, n_samples)

    # Fit polynomials (RANSAC)
    yellow_coeffs = ransac_polyfit(
        yellow_pts,
        degree=poly_degree,
        threshold=ransac_distance_threshold,
        iters=ransac_max_iterations,
    )
    white_coeffs = ransac_polyfit(
        white_pts,
        degree=poly_degree,
        threshold=ransac_distance_threshold,
        iters=ransac_max_iterations,
    )

    y_yellow = d_yellow = None
    y_white = d_white = None

    if yellow_coeffs is not None:
        y_yellow = np.polyval(yellow_coeffs, xs)
        d_yellow = np.polyval(np.polyder(yellow_coeffs), xs)

    if white_coeffs is not None:
        y_white = np.polyval(white_coeffs, xs)
        d_white = np.polyval(np.polyder(white_coeffs), xs)

    # Current mode:
    # y_yellow = None  # ignore white line for now

    # Case A: both lanes present
    #  if y_yellow is not None and y_white is not None:
    #      yx_s, yy_s = shift_poly_curve(xs, y_yellow, d_yellow, +half_width)
    #      wx_s, wy_s = shift_poly_curve(xs, y_white, d_white, -half_width)
    #      center_x = 0.5 * (yx_s + wx_s)
    #      center_y = 0.5 * (yy_s + wy_s)
    #      return list(zip(center_x, center_y))

    # Case B: rely on yellow
    #  yellow_pts_threshold = 20
    if y_white is None or len(yellow_pts) >= yellow_pts_threshold:
        cx, cy = shift_poly_curve(xs, y_yellow, d_yellow, +half_width)
        smoothed = decide_and_smooth(cx, cy, yellow_coeffs, traj_buffer)
        cx_s, cy_s = smoothed
        return list(zip(cx_s, cy_s))

    # Case B: rely on white
    if y_yellow is None or len(yellow_pts) < yellow_pts_threshold:
        print("Relying on white line for centerline")
        cx, cy = shift_poly_curve(xs, y_white, d_white, -half_width)
        smoothed = decide_and_smooth(cx, cy, white_coeffs, traj_buffer)
        cx_s, cy_s = smoothed
        return list(zip(cx_s, cy_s))

    # Case D: nothing detected ----
    return list(zip(xs, np.zeros_like(xs)))
