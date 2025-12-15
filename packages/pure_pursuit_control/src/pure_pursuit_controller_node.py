#!/usr/bin/env python3
import math
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import (
    Twist2DStamped,
    LanePose,
    WheelsCmdStamped,
    BoolStamped,
    FSMState,
    StopLineReading,
)
from nav_msgs.msg import Path

from pure_pursuit_control.include.pure_pursuit_controller.controller import (
    LaneController,
)

from pure_pursuit_control.include.pure_pursuit_controller.pure_pursuit_controller import (
    PurePursuitController,
)


class PurePursuitControllerNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(PurePursuitControllerNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.params["~lookahead_distance"] = DTParam(
            "~lookahead_distance",
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=1.0,
        )
        self.params["~kp_steering"] = DTParam(
            "~kp_steering", param_type=ParamType.FLOAT, min_value=0.0, max_value=5.0
        )
        self.params["~v_bar"] = DTParam(
            "~v_bar", param_type=ParamType.FLOAT, min_value=0.0, max_value=5.0
        )
        self.params["~stop_line_slowdown"] = rospy.get_param(
            "~stop_line_slowdown", None
        )

        # FIXME: can we get duckie width another way?
        self.params["~chassis_width"] = rospy.get_param(
            "~chassis_width", min_value=0.1, max_value=1.0
        )

        # Need to create controller object before updating parameters, otherwise it will fail
        #  self.controller = LaneController(self.params)
        self.controller = PurePursuitController(self.params)
        # self.updateParameters() # TODO: This needs be replaced by the new DTROS callback when it is implemented

        # Initialize variables
        self.stop_line_distance = None
        self.stop_line_detected = False
        self.at_stop_line = False
        self.obstacle_stop_line_distance = None
        self.obstacle_stop_line_detected = False
        self.at_obstacle_stop_line = False

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher(
            "~car_cmd", Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL
        )

        # Construct subscribers
        self.sub_trajectory = rospy.Subscriber(
            "~trajectory", Path, self.cbTrajectory, queue_size=1
        )
        self.sub_wheels_cmd_executed = rospy.Subscriber(
            "~wheels_cmd", WheelsCmdStamped, self.cbWheelsCmdExecuted, queue_size=1
        )
        self.sub_stop_line = rospy.Subscriber(
            "~stop_line_reading", StopLineReading, self.cbStopLineReading, queue_size=1
        )
        self.sub_obstacle_stop_line = rospy.Subscriber(
            "~obstacle_distance_reading",
            StopLineReading,
            self.cbObstacleStopLineReading,
            queue_size=1,
        )

        self.log("Pure Pursuit Controller Node Initialized!")

    def cbTrajectory(self, path_msg):
        self.trajectory = path_msg.poses
        self.path_points = [
            (msg.pose.position.x, msg.pose.position.y) for msg in path_msg.poses
        ]
        self.last_found_index = 0

        self.cbParametersChanged()

        # self.log(f"pure pursuit trajectory points: {len(self.path_points)}")

        self.computeControlAction(path_msg)

    def computeControlAction(self, path_msg):
        """
        Compute Control using line-circle intersection algorithm as descbribed in
        https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit
        """
        if self.at_stop_line or self.at_obstacle_stop_line:
            self.stopControl()
            return

        v, omega = self.controller.compute_control_action(self.path_points)

        # FIXME: reduce speed if stopline is near (or if corner)
        if self.stop_line_detected and (self.stop_line_distance is not None):
            slowdown_start = self.params[
                "~stop_line_slowdown"
            ].value  # FIXME: change to be within 2xlookahead distance?
            if self.stop_line_distance < slowdown_start:
                scale = max(0.0, self.stop_line_distance / slowdown_start)
                v = scale * v

        # 3. Update
        car_control_msg = Twist2DStamped()
        car_control_msg.header = path_msg.header
        car_control_msg.v = v
        car_control_msg.omega = omega
        self.pub_car_cmd.publish(car_control_msg)

    def stopControl(self):
        car_control_msg = Twist2DStamped()
        car_control_msg.header.stamp = rospy.Time.now()
        car_control_msg.v = 0.0
        car_control_msg.omega = 0.0
        self.pub_car_cmd.publish(car_control_msg)

    def cbObstacleStopLineReading(self, msg):
        """
        Callback storing the current obstacle distance, if detected.

        Args:
            msg(:obj:`StopLineReading`): Message containing information about the virtual obstacle stopline.
        """
        self.obstacle_stop_line_distance = np.sqrt(
            msg.stop_pose.x**2 + msg.stop_pose.y**2
        )
        self.obstacle_stop_line_detected = msg.stop_line_detected
        self.at_stop_line = msg.at_stop_line
        if not self.obstacle_stop_line_detected:
            self.obstacle_stop_line_distance = None

    def cbStopLineReading(self, msg):
        """Callback storing current distance to the next stopline, if one is detected.

        Args:
            msg (:obj:`StopLineReading`): Message containing information about the next stop line.
        """
        self.stop_line_distance = -msg.stop_pose.x
        self.stop_line_detected = msg.stop_line_detected
        self.at_obstacle_stop_line = msg.at_stop_line
        if not self.stop_line_detected:
            self.stop_line_distance = None

    def cbWheelsCmdExecuted(self, msg_wheels_cmd):
        """Callback that reports if the requested control action was executed.

        Args:
            msg_wheels_cmd (:obj:`WheelsCmdStamped`): Executed wheel commands
        """
        self.wheels_cmd_executed = msg_wheels_cmd

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)

    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    pure_pursuit_controller_node = PurePursuitControllerNode(
        node_name="pure_pursuit_controller_node"
    )
    # Keep it spinning
    rospy.spin()
