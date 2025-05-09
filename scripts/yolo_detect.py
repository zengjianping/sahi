import os, sys, math, copy
import glob, time, json
import argparse, cv2
import numpy as np
import pickle
from easydict import EasyDict as edict
from ultralytics import YOLO
from xml.dom import minidom
import xml.etree.ElementTree as ET

work_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, work_dir)

from sahi import AutoDetectionModel
from sahi import predict
from sahi.prediction import ObjectPrediction
from sahi.utils.file import list_files
from sahi.utils.cv import IMAGE_EXTENSIONS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='datas/models/yolo11n.pt')
    parser.add_argument('--input_path', type=str, default='datas/test_videos/multi02.avi')
    parser.add_argument('--config_file', type=str, default='configs/yolo_detect_config.json')
    args = parser.parse_args()
    return args

def yolo_detect_object(input_path, model_path, config_file):
    params = edict(json.loads(open(config_file).read()))
    input_dir = os.path.dirname(input_path)
    input_name = os.path.splitext(os.path.basename(input_path))[0] + '_results'
    class_ids = list(params.class_map.values())
    class_names = dict([(val,key) for key, val in params.class_map.items()])
    
    if params.use_sahi:
        results = predict.predict(
            model_type="ultralytics",
            model_path=model_path,
            model_device='cuda:0',
            model_confidence_threshold=params.thres_conf,
            model_category_mapping=None,
            model_category_remapping=None,
            source=input_path,
            no_standard_prediction=False,
            no_sliced_prediction=False,
            image_size=params.slice_size,
            slice_height=params.slice_size,
            slice_width=params.slice_size,
            overlap_height_ratio=params.overlap_ratio,
            overlap_width_ratio=params.overlap_ratio,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=params.thres_iou,
            postprocess_class_agnostic=False,
            project=input_dir,
            name=input_name,
            view_video=params.show_video,
            frame_skip_interval=params.frame_skip_interval,
            visual_hide_labels=not params.show_label,
            visual_hide_conf=not params.show_conf,
            export_pickle=True,
            verbose=2,
            return_dict=True
        )
        
        result_dir = results['export_dir']
        pickle_dir = os.path.join(result_dir, 'pickles')
        pickle_files = glob.glob(os.path.join(pickle_dir, '*.pickle'))
        pickle_files.sort()
        
        label_dir = os.path.join(result_dir, 'labels')
        os.makedirs(label_dir, exist_ok=True)
        
        for pickle_file in pickle_files:
            file_name = os.path.splitext(os.path.basename(pickle_file))[0]
            label_file = os.path.join(label_dir, f'{file_name}.txt')
            pickle_data = pickle.load(open(pickle_file,'rb'))
            prediction_list = pickle_data['predictions']
            image_width, image_height = pickle_data['image_size']
            wfile = open(label_file, 'w', encoding='utf-8')
            for idx in range(len(prediction_list)):
                prediction:ObjectPrediction = prediction_list[idx]
                cat_id = prediction.category.id
                if not class_ids or cat_id in class_ids:
                    box_x = (prediction.bbox.minx + prediction.bbox.maxx) / 2 / image_width
                    box_y = (prediction.bbox.miny + prediction.bbox.maxy) / 2 / image_height
                    box_w = (prediction.bbox.maxx - prediction.bbox.minx) / image_width
                    box_h = (prediction.bbox.maxy - prediction.bbox.miny) / image_height
                    score = prediction.score.value
                    line = '{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(cat_id,
                        box_x, box_y, box_w, box_h, score)
                    #box_x1 = prediction.bbox.minx
                    #box_y1 = prediction.bbox.miny 
                    #box_x2 = prediction.bbox.maxx
                    #box_y2 = prediction.bbox.maxy
                    #score = prediction.score.value
                    #line = '{} {:.3f} {:.3f} {:.3f} {:.3f} {:.6f}\n'.format(cat_id,
                    #    box_x1, box_y1, box_x2, box_y2, score)
                    wfile.write(line)
            wfile.close()

    else:
        model = YOLO(model_path)

        results = model(input_path, stream=False, project=input_dir, name=input_name,
            classes=class_ids, imgsz=params.image_size, conf=params.thres_conf,
            iou=params.thres_iou, max_det=params.max_det, save=params.save_image,
            save_txt=params.save_txt, save_conf=params.save_conf, show=params.show_image,
            show_labels=params.show_label, show_conf=params.show_conf)
        result_dir = results[0].save_dir
    
    if os.path.isdir(input_path) or os.path.splitext(input_path)[1] in IMAGE_EXTENSIONS:
        if os.path.isdir(input_path):
            image_files = list_files(input_path, IMAGE_EXTENSIONS)
        else:
            image_files = [input_path]

        yolo_label_dir = os.path.join(result_dir, 'labels')
        xml_label_dir = os.path.join(result_dir, 'xmls')
        os.makedirs(xml_label_dir, exist_ok=True)

        for image_file in image_files:
            image_name = os.path.splitext(os.path.basename(image_file))[0]
            yolo_label_file = os.path.join(yolo_label_dir, f'{image_name}.txt')
            if os.path.isfile(yolo_label_file):
                lines = open(yolo_label_file, 'r', encoding='utf-8').readlines()
                detections = list()
                for line in lines:
                    elems = [float(x) for x in line.strip().split(' ')]
                    detections.append([int(elems[0]), *elems[1:6]])
                create_voc_xml(image_file, detections, class_names, xml_label_dir)
    print(f'Results saved in directory: {result_dir}')

    return results

def yolo_to_voc(x_center, y_center, w, h, img_width, img_height):
    """将YOLO格式的归一化坐标转换为VOC格式的绝对坐标"""
    x_center *= img_width
    w *= img_width
    y_center *= img_height
    h *= img_height
    xmin = int(round((x_center - w / 2)))
    ymin = int(round(y_center - h / 2))
    xmax = int(round(x_center + w / 2))
    ymax = int(round(y_center + h / 2))
    return xmin, ymin, xmax, ymax

def create_voc_xml(image_path, detections, class_names, output_dir):
    """创建VOC格式的XML文件"""
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # 创建XML根结构
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = os.path.basename(output_dir)
    ET.SubElement(root, "filename").text = os.path.basename(image_path)
    #ET.SubElement(root, "path").text = image_path

    # 图像尺寸信息
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = "3"  # 假设为RGB图像
    k=0
    # 添加检测目标
    for det in detections:
        class_id = int(det[0])
        confidence = float(det[5])
        x_center, y_center, w, h = det[1:5]

        # 转换坐标
        xmin, ymin, xmax, ymax = yolo_to_voc(x_center, y_center, w, h, img_width, img_height)
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 250), 1)
        label = f" {class_names[class_id]} Conf: {confidence:.2f}"
        #cv2.putText(img, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        #cv2.imshow('image', img)
        #cv2.waitKey(500)
        area = (xmax - xmin) * (ymax - ymin)
        #if area < 200 :#'''or class_names[class_id] == 'person''''':
        #    continue
        #    print("area=%d" % area)
        k=k+1
        # 创建对象节点
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_names[class_id]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        #ET.SubElement(obj, "confidence").text = f"{confidence:.4f}"

        # 边界框
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    if k >= 0 :
        # 美化输出格式
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

        # 保存文件
        xml_filename = os.path.splitext(os.path.basename(image_path))[0] + ".xml"
        xml_path = os.path.join(output_dir, xml_filename)
        with open(xml_path, "w") as f:
            f.write(xml_str)

def main(args):
    yolo_detect_object(args.input_path, args.model_path, args.config_file)
    return True

if __name__ == '__main__':
    args = parse_args()
    main(args)

