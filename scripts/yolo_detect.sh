#!/bin/bash

CONFIG_FILE="configs/yolo_detect_config.json"
#MODEL_PATH="datas/models/yolo11s.pt"
#MODEL_PATH="datas/models/kst_vis_yolo10m_i0640.pt"
MODEL_PATH="datas/models/kst_ir_yolo10m_v3.pt"
#INPUT_PATH="datas/test_videos/multi02.avi"
#INPUT_PATH="datas/test_videos/airport_nanning_d02_vis_01_unfold.mp4"
#INPUT_PATH="datas/test_images/small-vehicles1.jpeg"
#INPUT_PATH="datas/test_images/airport_nanning_d02_vis_01_unfold"
INPUT_PATH="datas/test_images/airport_nanning_d02_ir_01"
#INPUT_PATH="datas/test_videos/airport_nanning_d02_ir_01.mp4"

python scripts/yolo_detect.py \
    --config_file "$CONFIG_FILE" \
    --model_path "$MODEL_PATH" \
    --input_path "$INPUT_PATH"


