#!/bin/bash

if [ -f /etc/os-release ] || [ -d /proc ]; then
    echo "Running on Linux system"
    
    # 啟動虛擬X Server
    if ! pgrep -x "Xvfb" > /dev/null; then
        Xvfb :0 -screen 0 1280x1024x24 &
        sleep 2
    fi
    export DISPLAY=:0
    
    # Isaac Sim conda Env
    source ~/isaacsim/setup_conda_env.sh    
    # 使用headless模式訓練
    python scripts/rl_games/train.py --task=Custom-Factory-PegInsert-RGBD-Camera-Direct-v0 --enable_cameras --headless
    
else
    # Windows系统
    echo "Running on Windows system"
    python scripts/rl_games/train.py --task=Custom-Factory-PegInsert-RGBD-Camera-Direct-v0 --enable_cameras
fi