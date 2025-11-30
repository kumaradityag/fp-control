#!/usr/bin/env python3

import numpy as np
import cv2
import rospy

from cv_bridge import CvBridge
from nav_msgs.msg import Path
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from duckietown_msgs.msg import SegmentList, Segment as SegmentMsg

from duckietown.dtros import DTROS, NodeType, TopicType
from dt_computer_vision.ground_projection.types import GroundPoint
from dt_computer_vision.ground_projection.rendering import (
    draw_grid_image,
    robot_to_image_frame,
)

from trajectory_planner.include import trajectory_generation

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

        # Parameters
        self.max_forward = rospy.get_param("~max_forward", 1.0)
        self.n_samples = rospy.get_param("~n_samples", 25)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 0.35)
        self.lane_width = rospy.get_param("~lane_width", 0.23)

        # debug grid params (match ground_projection)
        self.grid_size = rospy.get_param("~grid_size", 4)
        self.scale = rospy.get_param("~scale", 1000)
        self.padding = rospy.get_param("~padding", 80)
        self.resolution = rospy.get_param("~resolution", 0.3)

        self.bridge = CvBridge()

        self.debug = True

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
        # Convert lane boundaries into an ordered centerline path
        yellow_pts, white_pts = self.get_lane_boundaries(msg)
        centerline_pts = trajectory_generation.compute_centerline(
            yellow_pts,
            white_pts,
            self.max_forward,
            self.n_samples,
            self.lane_width,
        )

        # publish trajectory
        path_msg = self.build_path_message(msg, centerline_pts)
        self.pub_path.publish(path_msg)

        # publish debug image
        # if self.pub_debug_img.anybody_listening():
        if self.debug:
            debug_img = self.render_debug(msg, centerline_pts)
            dbg = self.bridge.cv2_to_compressed_imgmsg(debug_img)
            dbg.header = msg.header
            self.pub_debug_img.publish(dbg)

    def get_lane_boundaries(self, seglist: SegmentList) -> Tuple[np.array, np.array]:
        """
        Convert segment list into yellow and white lane boundaries and 
        remove white lines on the left side
        """
        yellow_pts = []
        white_pts = []

        remove_seglist_idx = []
        for idx, seg in enumerate(seglist.segments):
            p1 = np.array([seg.points[0].x, seg.points[0].y])
            p2 = np.array([seg.points[1].x, seg.points[1].y])

            if seg.color == SegmentMsg.YELLOW:
                yellow_pts += [p1, p2]
            elif seg.color == SegmentMsg.WHITE:
                if (p1[1] > 0.0) or (p2[1] > 0.0): # TODO: ignore white lines at the left of yellow lines
                    remove_seglist_idx.append(idx)  # ignore white lines on left side
                    continue
                white_pts += [p1, p2]

        # Remove unwanted segments
        for idx in sorted(remove_seglist_idx, reverse=True):
            del seglist.segments[idx]

        yellow_pts = np.array(yellow_pts)
        white_pts = np.array(white_pts)

        return yellow_pts, white_pts

    def build_path_message(self, seglist: SegmentList, centerline: 
                           List[Tuple[float, float]]) -> Path:
        if len(centerline) == 0:
            return Path()

        path_msg = Path()
        path_msg.header = seglist.header

        for x, y in centerline:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            path_msg.poses.append(pose)

        return path_msg

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
            grid_size=self.grid_size,
            scale=self.scale,
            s_padding=self.padding,
            resolution=self.resolution,
            start_x=0.0,
        )

        resolution = (self.resolution, self.resolution)
        grid_x = grid_y = self.grid_size
        size_u, size_v = size

        s = max(size_u, size_v) / self.scale
        padding_px = int(self.padding * s)

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
            mx = 0.5 * (p1.x + p2.x)
            my = 0.5 * (p1.y + p2.y)
            midpoint = GroundPoint(mx, my)

            # Convert midpoint to debug-image pixel frame
            um, vm = robot_to_image_frame(
                midpoint, resolution, (origin_u, origin_v), (cell_x, cell_y)
            )

            # Ground-frame normal vector (already computed in earlier node)
            nx = seg.normal.x
            ny = seg.normal.y

            # Scale the vector for visibility on the debug image
            normal_scale = 0.10  # meters → length of arrow
            endp = GroundPoint(mx + nx * normal_scale, my + ny * normal_scale)

            ue, ve = robot_to_image_frame(
                endp, resolution, (origin_u, origin_v), (cell_x, cell_y)
            )

            # Draw a blue arrow
            cv2.arrowedLine(
                img,
                (um, vm),
                (ue, ve),
                (255, 0, 0),  # BLUE
                thickness=max(1, int(4 * s)),
                tipLength=0.3,
            )

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
