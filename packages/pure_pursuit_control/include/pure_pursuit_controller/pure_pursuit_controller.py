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

        # Need to recompute the lookahead distance in case we the goal point 
        # used is not one of the intersection point at the original lookahead 
        # distance
        L_d = math.sqrt(dx**2 + dy**2) 
        #  if L_d < 0.01:
        #      return v_bar, 0.0

        #  v = v_bar
        width = self.parameters["~width"].value
        #  v_max = self.parameters["~v_max"].value
        omega_factor = self.parameters["~omega_factor"].value
        v_bar_min = self.parameters["~v_bar_min"].value
        v_bar_max = self.parameters["~v_bar_max"].value


        v = v_bar
        R = np.abs(L_d / (2.0 * np.sin(turn_error)))
        #  scaling_factor = 1.0 / (1.0 + R)
        #  scaling_factor = 1.0 / (1.0 + R) if R < 
        v2 = np.clip(v_bar * R, v_bar_min, v_bar_max)
        print(R, v, v2)

        omega = kp * turn_error
        omega2 = omega_factor * (width * np.sin(turn_error) * v) / L_d
        #  print(omega, omega2)
        return v2, omega2
