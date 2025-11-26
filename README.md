# Duckietown Fall 2025 - Final Project - Control

We aim to do map-free lane following in this project. We plan to use a Pure Pursuit controller to achieve this.

## Running this thing

I assume you have a duckiebot setup, know how to start a virtual duckiebot, and know your way around the duckiematrix.

This repo follows the `template-ros` provided by Duckietown. If the requirements listed above are complete, you should be able to build the image specified by this repo and run things on your duckiebot (virtual or not).

The steps assumes a virtual duckiebot called `vbot` running. Change this to your duckiebot name please.

Follow these steps:
```bash
git clone git@github.com:kumaradityag/fp-control
```

You will need 4 terminals to view everything needed. In all terminals:
```bash
cd fp-control
```

Then in terminal 1, launch the duckiematrix. There are two map options within this repo - `straight` and `loop`:
```
# Check if the duckiebot is active
dts fleet discover
dts matrix run --standalone --map ./assets/duckiematrix/map/straight/
```

In terminal 2:
```bash
dts matrix attach vbot map_0/vehicle_0
dts devel build -H vbot -f
dts devel run -H vbot -L lane-following
```

In terminal 3:
```bash
dts gui vbot
# Wait for the entrypoint - inside it run:
rqt_image_view
```

In terminal 4:
```bash
ssh duckie@vbot.local
# After logging into your duckiebot:
docker exec -it ros-interface bash
```
Terminal 4 will allow you to check what is happening with ROS on your duckiebot. For example you'll be able to run commands like:
```bash
rostopic list
```

## Current Progress
@kumaradityag: I have written a node to compute a basic trajectory and publish it to `/vbot/trajectory_planner_node/trajectory`. A debug image should also be published with the trajectory in <span style="color:red">red</span>. The debug image topic is `/vbot/trajectory_planner_node/debug/trajectory_image/compressed`. Use `rqt_image_view` to view it. The trajectory is *not* great now. Working on fixing it :)
@yukikongju: I have started writting the pure pursuit package in the branch feature/pure_pursuit. Pure Pursuit is not implemented yet, but working on it:)
