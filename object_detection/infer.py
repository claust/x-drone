import datetime

import os, sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # object_detection root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

ROOT_YOLO = ROOT / '../../yolov9'
if str(ROOT_YOLO) not in sys.path:
    sys.path.append(str(ROOT_YOLO))  # add ROOT to PATH

def get_path():
    return ROOT.resolve()

def get_yolo_path():
    return ROOT_YOLO.resolve()

from detect import run

def infer(
        weights=ROOT / 'weights/best_striped.pt',  # model path or triton URL
        source=ROOT_YOLO / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT_YOLO / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'data/runs',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    run(weights, source, data, imgsz, conf_thres, iou_thres, max_det, device, view_img, save_txt, save_conf, save_crop,
        nosave, classes, agnostic_nms, augment, visualize, update, project, name, exist_ok, line_thickness, hide_labels,
        hide_conf, half, dnn, vid_stride)
    