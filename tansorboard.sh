#!/bin/bash

# 設定基本日誌目錄
BASE_LOG_DIR="logs/rl_games/Factory/visual_factory_multi_camera/summaries"

# 檢查目錄是否存在
if [ ! -d "$BASE_LOG_DIR" ]; then
    echo "錯誤：日誌目錄 $BASE_LOG_DIR 不存在"
    exit 1
fi

# 找出最新的 events.out.tfevents 檔案
LATEST_EVENT=$(find $BASE_LOG_DIR -name "events.out.tfevents*" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$LATEST_EVENT" ]; then
    echo "找不到任何 events.out.tfevents 檔案，將使用整個日誌目錄"
    echo "啟動 TensorBoard..."
    tensorboard --logdir=$BASE_LOG_DIR --port 6009
else
    # 取得包含最新事件檔案的目錄
    EVENT_DIR=$(dirname "$LATEST_EVENT")
    echo "找到最新的事件檔案：$LATEST_EVENT"
    echo "使用目錄：$EVENT_DIR"
    echo "啟動 TensorBoard..."
    
    # 使用包含最新事件檔案的目錄啟動 TensorBoard
    tensorboard --logdir=$EVENT_DIR --port 6009
fi