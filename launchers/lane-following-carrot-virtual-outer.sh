#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

# launching app
dt-exec roslaunch --wait duckietown_demos lane_following_carrot.launch pp_param_file_name:=virtual_outer traj_param_file:=outer


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
