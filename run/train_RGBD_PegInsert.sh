#!/bin/bash

# Start a virtual X server if not already running
if ! pgrep -x "Xvfb" > /dev/null; then
    Xvfb :0 -screen 0 1280x1024x24 &
    sleep 2  # Give it time to start
fi

# Set display variable
export DISPLAY=:0

# Run the training script with correct arguments
# Removed the unrecognized --sim_device argument
python scripts/rl_games/train.py --task=Custom-Factory-PegInsert-RGBD-Camera-Direct-v0 --enable_cameras --headless