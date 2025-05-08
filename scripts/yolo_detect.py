import os, sys, math, copy
import glob, time, json
import argparse, cv2
import numpy as np
import pickle
from easydict import EasyDict as edict
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi import predict
from sahi.prediction import ObjectPrediction

work_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, work_dir)


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
    
    if params.use_sahi:
        #cap = cv2.VideoCapture(input_path)
        #frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        #frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

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
        
        pickle_dir = os.path.join(results['export_dir'], 'pickles')
        pickle_files = glob.glob(os.path.join(pickle_dir, '*.pickle'))
        pickle_files.sort()
        
        label_dir = os.path.join(results['export_dir'], 'labels')
        os.makedirs(label_dir, exist_ok=True)
        
        for pickle_file in pickle_files:
            file_name = os.path.splitext(os.path.basename(pickle_file))[0]
            label_file = os.path.join(label_dir, f'{file_name}.txt')
            prediction_list = pickle.load(open(pickle_file,'rb'))
            wfile = open(label_file, 'w', encoding='utf-8')
            for idx in range(len(prediction_list)):
                prediction:ObjectPrediction = prediction_list[idx]
                cat_id = prediction.category.id
                if not params.classes or cat_id in params.classes:
                    #box_x = (prediction.bbox.minx + prediction.bbox.maxx) / 2 / frame_width
                    #box_y = (prediction.bbox.miny + prediction.bbox.maxy) / 2 / frame_height
                    #box_w = (prediction.bbox.maxx - prediction.bbox.minx) / frame_width
                    #box_h = (prediction.bbox.maxy - prediction.bbox.miny) / frame_height
                    #score = prediction.score.value
                    #line = '{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(cat_id,
                    #    box_x, box_y, box_w, box_h, score)
                    box_x1 = prediction.bbox.minx
                    box_y1 = prediction.bbox.miny 
                    box_x2 = prediction.bbox.maxx
                    box_y2 = prediction.bbox.maxy
                    score = prediction.score.value
                    line = '{} {:.3f} {:.3f} {:.3f} {:.3f} {:.6f}\n'.format(cat_id,
                        box_x1, box_y1, box_x2, box_y2, score)
                    wfile.write(line)
            wfile.close()

    else:
        model = YOLO(model_path)

        results = model(input_path, stream=False, project=input_dir, name=input_name,
            classes=params.classes, imgsz=params.image_size, conf=params.thres_conf,
            iou=params.thres_iou, max_det=params.max_det, save=params.save_image,
            save_txt=params.save_txt, save_conf=params.save_conf, show=params.show_image,
            show_labels=params.show_label, show_conf=params.show_conf)

    return results

def main(args):
    yolo_detect_object(args.input_path, args.model_path, args.config_file)
    return True

if __name__ == '__main__':
    args = parse_args()
    main(args)

