#! /usr/bin/env python
# -*- coding: utf-8 -*-
from easydict import EasyDict as edict
import yolov4_config
import sys
import os

class_label_path  = '../Yolov5_tf/class_label.txt'
class_name_list = yolov4_config.Customer_DATA["CLASSES"]
ftmp = open(class_label_path, 'w')
for idx, class_name in enumerate(class_name_list):
    ftmp.write(class_name)
    if idx != len(class_name_list)-1:
        ftmp.write('\n')
ftmp.close()

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# YOLO options
__C.YOLO = edict()


# Set the class name
__C.YOLO.NET_TYPE = 'mobilenetv2' # 'darknet53' 'mobilenetv2' 'mobilenetv3' 'mobilenetv3_small'
__C.YOLO.CLASSES = class_label_path #'/home/chenp/Yolov5_tf/class_label.txt'
__C.YOLO.ANCHORS = 'data/anchors/basline_anchors.txt' # yolov3/5 : yolo_anchors.txt; yolov4 : yolov4_anchors.txt
__C.YOLO.MOVING_AVE_DECAY = 0.9995
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.STRIDES_TINY = [16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5
__C.YOLO.UPSAMPLE_METHOD = 'resize'

__C.YOLO.WIDTH_SCALE_V5 = 0.50 # yolov5 small:0.50 / middle:0.75 / large:1.00 / extend:1.25
__C.YOLO.DEPTH_SCALE_V5 = 0.33 # yolov5 small:0.33(1/3) / middle:0.67(2/3) / large:1.00 / extend:1.33(4/3)

__C.YOLO.ORIGINAL_WEIGHT = 'checkpoint/yolov3_coco.ckpt'
__C.YOLO.DEMO_WEIGHT = 'checkpoint/yolov3_coco_demo.ckpt'


# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = '../Yolov5_tf/train_annotation.txt'
__C.TRAIN.BATCH_SIZE = 2 if __C.YOLO.NET_TYPE == 'darknet53' else 32
__C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608] if not 'mobilenetv3' in __C.YOLO.NET_TYPE else [416]
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LEARN_RATE_INIT = 1e-4
__C.TRAIN.LEARN_RATE_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 10
__C.TRAIN.FISRT_STAGE_EPOCHS = 100
__C.TRAIN.SECOND_STAGE_EPOCHS = 1000

#restore from the best ckpt
f_in = open('./best-ckpt-path', 'r')
ckpt_file = f_in.readline()
f_in.close()
ckpt_file = ckpt_file.strip()
print('ckpt_file confirm: ', ckpt_file)
#input()
if not os.path.exists(ckpt_file + '.index'):
    print('freeze_ckpt_to_pb ckpt_file=', ckpt_file, ' not exist')
    sys.exit()
__C.TRAIN.INITIAL_WEIGHT = ckpt_file #'checkpoint/yolov4_coco.ckpt' #'ckpts/yolov3_test-loss=24.0873.ckpt-27'

__C.TRAIN.CKPT_PATH = 'checkpoint-v2' #'ckpts'


# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = '../Yolov5_tf/test_annotation.txt'
__C.TEST.BATCH_SIZE = 1
__C.TEST.INPUT_SIZE = 416
__C.TEST.DATA_AUG = False
__C.TEST.WRITE_IMAGE = True
__C.TEST.WRITE_IMAGE_PATH = 'detection_res/'
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE = 'checkpoint/social_yolov3_test-loss=3.3218.ckpt-51'
__C.TEST.SHOW_LABEL = True
__C.TEST.SCORE_THRESHOLD = 0.3
__C.TEST.IOU_THRESHOLD = 0.45
