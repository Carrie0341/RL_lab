#!/bin/bash

# 更精確地檢測Windows環境
if [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == MSYS* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win"* ]]; then
    # Windows系统 (包括MINGW/Git Bash環境)
    echo "Running on Windows system"
    python scripts/rl_games/train.py --task=Custom-Factory-PegInsert-RGBD-Camera-Direct-v0 --enable_cameras
elif [ -f /etc/os-release ] || [ -d /proc ]; then
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
    python scripts/rl_games/train.py --task=Custom-Factory-PegInsert-RGBD-Camera-Direct-v0 --enable_cameras --headless --num_envs=128
else
    # 其他系統
    echo "Unknown operating system"
    python scripts/rl_games/train.py --task=Custom-Factory-PegInsert-RGBD-Camera-Direct-v0 --enable_cameras --headless  --num_envs=128
fi