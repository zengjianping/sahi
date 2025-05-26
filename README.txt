
一、程序执行
1. 执行脚本“scripts/yolo_detect.sh”启动切片检测程序。
2. SHELL脚本“scripts/yolo_detect.sh”实际执行python文件“scripts/yolo_detect.py”，有三个参数。
3. 参数“MODEL_PATH”设置模型文件，参数“INPUT_PATH”设置待处理图片目录或视频文件，参数“CONFIG_FILE”设置切片检测算法配置文件。
4. 如果算法参数文件中show_image置为true，则显示结果图像或视频，启动时会停止在第一帧图像，可以按空格键切换播放和暂停，可以使用f/F键单步前进。

二、结果输出
1. 处理结果保存在*_results目录中，*是处理的图像目录或视频文件名称，如果*_results目录存在，则保存在目录*_results2中，以此类推。
2. 结果目录中，子目录xmls保存xml格式的检测框文件，子目录labels保存yolo格式的检测框文件，visuals保存结果图像文件。

三、参数说明
切片检测算法配置文件“configs/yolo_detect_config.json”的参数说明如下。

"class_map": 检测类型名称与ID的映射字典，默认值为{"plane":0,"car":1,"person":2}
"image_size": 模型检测的图像尺寸，默认值为640
"thres_conf": 检测置信度阈值，默认值为0.3,
"thres_iou": 检测框非最大抑制阈值，默认值为0.7,
"max_det": 最大检测框的数目，默认值为100,
"save_image": 是否保存结果图像或视频，默认值为true,
"show_image": 是否显示结果图像或视频，默认值为false
"use_sahi": 是否使用切片检测，默认值为true
"slice_size": 切片图像尺寸，默认值为640
"grid_width": 切片网格宽度，默认值为12
"grid_height": 切片网格高度，默认值为1
"zoom_width": 将原图缩放到尺寸(zoom_width,zoom_height)再做检测，zoom_width和zoom_height都为0时使用原图尺寸，
"zoom_height": zoom_width和zoom_height有一个大于0另一个小于0,则根据高度或宽度做等比缩放
"overlap_ratio": grid_width和grid_height都为0，使用切片图像之间的重叠比率计算切片网格位置，默认值为0.2
"frame_skip_interval": 处理视频和图片序列时的跳帧数目，默认值为0

