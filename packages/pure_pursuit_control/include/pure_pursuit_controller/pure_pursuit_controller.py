import numpy as np
import math

from pure_pursuit_control.include.pure_pursuit_controller.goal import (
    find_goal_point, sgn
)

class PurePursuitController:

    def __init__(self, parameters):
        self.parameters = parameters
        self.trajectory = []
        self.path_points = []
        self.last_found_index = 0
        self.current_pos = [0.0, 0.0]
        self.current_heading = 0.0
        #  self.num_frames = 400  # FIXME needed?


    def update_parameters(self, parameters):
        self.parameters = parameters

    def compute_control_action(self, path_points):
        lookahead_distance = self.parameters["~lookahead_distance"].value
        v_bar = self.parameters["~v_bar"].value
        kp_steering = self.parameters["~kp_steering"].value
        kp_linear = self.parameters["~kp_linear"].value
        width = self.parameters["~chassis_width"].value

        # 1. Find goal points
        goal_point, last_found_index = find_goal_point(
            path_points,
            self.current_pos,
            lookahead_distance,
            self.last_found_index,
        )

        # 2. Compute control - compute turn error
        dx, dy = (
            goal_point[0] - self.current_pos[0],
            goal_point[1] - self.current_pos[1],
        )
        abs_target_angle = math.atan2(dy, dx) * 180 / math.pi
        if abs_target_angle < 0:
            abs_target_angle += 360

        turn_error = abs_target_angle - self.current_heading
        if turn_error > 180 or turn_error < -180:
            turn_error = -1 * sgn(turn_error) * (360 - abs(turn_error))
        turn_error_rad = np.deg2rad(turn_error)

        # 3.1. Compute Linear velocity adjusted with curvature => k = 1 / R
        #  v = v_bar # Naive
        R = np.abs(lookahead_distance / (2 * np.sin(turn_error_rad)))
        scaling_factor = 1.0 / (1.0 + kp_linear / R) if R > 0.0 else 1.0
        v = v_bar * scaling_factor
        #  v = np.sqrt(v_bar * R)

        # 3.2. Compute Linear velocity adjusted with curvature
        #  omega = kp * turn_error_rad # from purdue
        omega = (width * np.sin(turn_error_rad)) / lookahead_distance * v

        return v, omega
