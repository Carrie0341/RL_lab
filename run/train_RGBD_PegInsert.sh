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
    python scripts/rl_games/train.py --task=Custom-Factory-PegInsert-RGBD-Camera-Direct-v0 --enable_cameras --headless --num_envs=64 --checkpoint="logs/rl_games/Factory_Multi_Camera/visual_factory_multi_camera_v5_2/nn/last_Factory_Multi_Camera_ep_1500_rew_214.95113.pth"
else
    # 其他系統
    echo "Unknown operating system"
    python scripts/rl_games/train.py --task=Custom-Factory-PegInsert-RGBD-Camera-Direct-v0 --enable_cameras --headless  --num_envs=64 --checkpoint="logs/rl_games/Factory_Multi_Camera/visual_factory_multi_camera_v5_2/nn/last_Factory_Multi_Camera_ep_1500_rew_214.95113.pth"
fi