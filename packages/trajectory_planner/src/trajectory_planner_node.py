#!/usr/bin/env python3

import numpy as np
import cv2
import rospy

from cv_bridge import CvBridge
from nav_msgs.msg import Path
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from duckietown_msgs.msg import SegmentList, Segment as SegmentMsg

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from dt_computer_vision.ground_projection.types import GroundPoint
from dt_computer_vision.ground_projection.rendering import (
    draw_grid_image,
    robot_to_image_frame,
)

from trajectory_planner.include import trajectory_generation
from trajectory_planner.include.buffer import TrajectoryBuffer

from typing import List, Tuple


class TrajectoryPlannerNode(DTROS):
    """
    Computes a centerline trajectory from projected lane segments
    and publishes a Path for pure pursuit control.

    Also publishes a debug visualization identical
    to the ground_projection debug output.
    """

    def __init__(self, node_name):
        super(TrajectoryPlannerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PLANNING,
        )

        # Parameters (auto-updating via DTParam)
        self.max_forward = DTParam(
            "~max_forward",
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=1.0,
        )

        self.n_samples = DTParam(
            "~n_samples",
            param_type=ParamType.INT,
            min_value=1,
            max_value=200,
        )

        self.poly_degree = DTParam(
            "~poly_degree",
            param_type=ParamType.INT,
            min_value=1,
            max_value=5,
        )
        self.ransac_max_iterations = DTParam(
            "~ransac_max_iterations",
            param_type=ParamType.INT,
            min_value=1,
            max_value=1000,
        )
        self.ransac_distance_threshold = DTParam(
            "~ransac_distance_threshold",
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=1.0,
        )

        self.lane_width = DTParam(
            "~lane_width",
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=1.0,
        )

        self.epsilon = DTParam(
            "~epsilon",
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=1.0,
        )
        self.buffer_size = DTParam(
            "~buffer_size", param_type=ParamType.INT, min_value=1, max_value=20
        )
        self.buffer_threshold = DTParam(
            "~buffer_threshold",
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=1.0,
        )
        self.buffer_smooth_alpha = DTParam(
            "~buffer_smooth_alpha",
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=1.0,
        )

        # Debug grid params
        self.grid_size = DTParam(
            "~grid_size", param_type=ParamType.INT, min_value=1, max_value=10
        )
        self.scale = DTParam(
            "~scale", param_type=ParamType.INT, min_value=100, max_value=5000
        )
        self.padding = DTParam(
            "~padding", param_type=ParamType.INT, min_value=0, max_value=300
        )
        self.resolution = DTParam(
            "~resolution", param_type=ParamType.FLOAT, min_value=0.01, max_value=1.0
        )

        self.bridge = CvBridge()

        self.debug = True

        self.traj_buffer = TrajectoryBuffer(
            max_size=self.buffer_size.value,
            change_threshold=self.buffer_threshold.value,
            smooth_alpha=self.buffer_smooth_alpha.value,
        )

        self.sub_segments = rospy.Subscriber(
            "~segments",
            SegmentList,
            self.cb_segments,
            queue_size=1,
        )

        self.pub_path = rospy.Publisher(
            "~trajectory",
            Path,
            queue_size=1,
        )

        self.pub_debug_img = rospy.Publisher(
            "~debug/trajectory_image/compressed",
            CompressedImage,
            queue_size=1,
        )

        self.loginfo("TrajectoryPlannerNode initialized.")

    # Main callback
    def cb_segments(self, msg: SegmentList):
        """
        Receive projected ground segments and compute a centerline.
        """
        path_msg, centerline_pts = self.compute_centerline_path(msg)

        # publish trajectory
        self.pub_path.publish(path_msg)

        # publish debug image
        # if self.pub_debug_img.anybody_listening():
        if self.debug:
            debug_img = self.render_debug(msg, centerline_pts)
            dbg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
            dbg.header = msg.header
            self.pub_debug_img.publish(dbg)

    # Centerline computation
    def compute_centerline_path(
        self, seglist: SegmentList
    ) -> Tuple[Path, List[Tuple[float, float]]]:
        """
        Convert lane boundaries into an ordered centerline path,
        then produce a nav_msgs/Path for pure pursuit.
        """

        yellow_pts = []
        white_pts = []

        yellow_normals = []
        white_normals = []

        for seg in seglist.segments:
            p1 = np.array([seg.points[0].x, seg.points[0].y])
            p2 = np.array([seg.points[1].x, seg.points[1].y])

            normal = np.array([seg.normal.x, seg.normal.y])

            if seg.color == SegmentMsg.YELLOW:
                yellow_pts += [p1, p2]
                yellow_normals += [normal, normal]
            elif seg.color == SegmentMsg.WHITE:
                white_pts += [p1, p2]
                white_normals += [normal, normal]

        if len(yellow_pts) < 2 or len(white_pts) < 2:
            return Path(), []

        yellow_pts = np.array(yellow_pts)
        white_pts = np.array(white_pts)
        yellow_normals = np.array(yellow_normals)
        white_normals = np.array(white_normals)

        centerline = trajectory_generation.compute_centerline(
            yellow_pts,
            white_pts,
            self.traj_buffer,
            self.max_forward.value,
            self.n_samples.value,
            self.lane_width.value,
            self.epsilon.value,
            self.poly_degree.value,
            self.ransac_max_iterations.value,
            self.ransac_distance_threshold.value,
        )

        # Build Path message
        path_msg = Path()
        path_msg.header = seglist.header

        for x, y in centerline:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            path_msg.poses.append(pose)

        return path_msg, centerline

    # Debug image
    def render_debug(
        self, seglist: SegmentList, trajectory_points: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Reproduce the ground_projection-style debug image,
        but overlay the computed trajectory centerline AND segment normals.
        """

        size = (600, 600)
        background = draw_grid_image(
            size=size,
            grid_size=self.grid_size.value,
            scale=self.scale.value,
            s_padding=self.padding.value,
            resolution=self.resolution.value,
            start_x=0.0,
        )

        resolution = (self.resolution.value, self.resolution.value)
        grid_x = grid_y = self.grid_size.value
        size_u, size_v = size

        s = max(size_u, size_v) / self.scale.value
        padding_px = int(self.padding.value * s)
        half_x = int(grid_x / 2)
        cell_x = int((size_u - 3 * padding_px) / grid_x)
        cell_y = int((size_v - 3 * padding_px) / grid_y)

        origin_u = 2 * padding_px + half_x * cell_x
        origin_v = size_v - 2 * padding_px

        img = background.copy()

        # ----------------------------------------------------------------------
        # Draw SEGMENTS and NORMALS
        # ----------------------------------------------------------------------
        for seg in seglist.segments:
            p1 = GroundPoint(seg.points[0].x, seg.points[0].y)
            p2 = GroundPoint(seg.points[1].x, seg.points[1].y)

            # pick color for the segment
            if seg.color == SegmentMsg.YELLOW:
                color = (0, 255, 255)
            elif seg.color == SegmentMsg.WHITE:
                color = (255, 255, 255)
            else:
                color = (0, 0, 255)

            # project endpoints
            u1, v1 = robot_to_image_frame(
                p1, resolution, (origin_u, origin_v), (cell_x, cell_y)
            )
            u2, v2 = robot_to_image_frame(
                p2, resolution, (origin_u, origin_v), (cell_x, cell_y)
            )

            # Draw the segment line
            cv2.line(img, (u1, v1), (u2, v2), color, max(1, int(6 * s)))

            # ------------------------------------------------------------------
            # Draw NORMAL (in BLUE)
            # ------------------------------------------------------------------
            # Compute midpoint in ground frame
            # mx = 0.5 * (p1.x + p2.x)
            # my = 0.5 * (p1.y + p2.y)
            # midpoint = GroundPoint(mx, my)

            # # Convert midpoint to debug-image pixel frame
            # um, vm = robot_to_image_frame(
            #     midpoint, resolution, (origin_u, origin_v), (cell_x, cell_y)
            # )

            # # Ground-frame normal vector (already computed in earlier node)
            # nx = seg.normal.x
            # ny = seg.normal.y

            # # Scale the vector for visibility on the debug image
            # normal_scale = 0.10  # meters → length of arrow
            # endp = GroundPoint(mx + nx * normal_scale, my + ny * normal_scale)

            # ue, ve = robot_to_image_frame(
            #     endp, resolution, (origin_u, origin_v), (cell_x, cell_y)
            # )

            # # Draw a blue arrow
            # cv2.arrowedLine(
            #     img,
            #     (um, vm),
            #     (ue, ve),
            #     (255, 0, 0),  # BLUE
            #     thickness=max(1, int(4 * s)),
            #     tipLength=0.3,
            # )

        # ----------------------------------------------------------------------
        # Draw CENTERLINE TRAJECTORY (red dots)
        # ----------------------------------------------------------------------
        for x, y in trajectory_points:
            gp = GroundPoint(x, y)
            u, v = robot_to_image_frame(
                gp, resolution, (origin_u, origin_v), (cell_x, cell_y)
            )
            cv2.circle(img, (u, v), int(8 * s), (0, 0, 255), -1)

        return img


if __name__ == "__main__":
    node = TrajectoryPlannerNode("trajectory_planner_node")
    rospy.spin()
