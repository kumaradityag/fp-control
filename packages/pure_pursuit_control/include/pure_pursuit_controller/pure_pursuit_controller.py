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
        kp = self.parameters["~kp_steering"].value

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
        abs_target_angle = math.atan2(dy, dx)
        turn_error = abs_target_angle - np.deg2rad(self.current_heading)

        #  L_d = math.sqrt(dx**2 + dy**2)
        #  if L_d < 0.01:
        #      return v_bar, 0.0

        v = v_bar
        omega = kp * turn_error

        return v, omega
