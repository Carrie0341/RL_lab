#!/bin/bash

# 检测操作系统类型
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux系统执行
    echo "Running on Linux system"
    
    # 启动虚拟X服务器（如果尚未运行）
    if ! pgrep -x "Xvfb" > /dev/null; then
        Xvfb :0 -screen 0 1280x1024x24 &
        sleep 2  # 给它启动的时间
    fi
    
    # 设置显示变量
    export DISPLAY=:0
    
    # 加载Isaac Sim conda环境
    source ~/isaacsim/setup_conda_env.sh    
    # 使用headless参数运行训练脚本
    python scripts/rl_games/train.py --task=Custom-Factory-PegInsert-RGBD-Camera-Direct-v0 --enable_cameras --headless
    
else
    # Windows系统
    echo "Running on Windows system"
    python scripts/rl_games/train.py --task=Custom-Factory-PegInsert-RGBD-Camera-Direct-v0 --enable_cameras
fi