import sys
sys.path.append(".")

import xml.etree.ElementTree as ET
import yolov4_config as cfg
import os
from tqdm import tqdm
import io
import json
import cv2
import get_time_util
import data_stream_status_machine

def parse_custom_annotation(data_path, anno_path, data_list_path):
    """
    phase pascal voc annotation, eg:[image_global_path xmin,ymin,xmax,ymax,cls_id]
    :param data_path: eg: VOC\VOCtrainval-2007\VOCdevkit\VOC2007
    :param anno_path: path to ann file
    :return: batch size of data set
    """
    if cfg.TRAIN["DATA_TYPE"] == "Customer":
        classes = cfg.Customer_DATA["CLASSES"]
        print('sds', type(classes), classes)
        input()
    else:
        print('not custom data.')
        input()
    image_num = 0
    folder_list = os.listdir(data_path)

    data_list_f = open(data_list_path, 'w')
    with open(anno_path, "w") as f:
        for folder in folder_list:
            class_id = classes.index(folder)

            sub_folder = data_path + '/' + folder
            sub_folder_list = os.listdir(sub_folder)

            for tp_f in sub_folder_list:
                sub_sub_folder = sub_folder + '/' + tp_f
                if not os.path.isdir(sub_sub_folder):
                    continue
                f_list = os.listdir(sub_sub_folder)
            
                for f_idx, file_tp in enumerate(f_list):
                    file_cur = sub_sub_folder + '/' + file_tp
                    filename, file_type = os.path.splitext(file_cur)
                    if file_type == '.json':
                        enable_save_img = False
                        need_check = False
                        print('file_cur', file_cur)
                        with io.open(file_cur, 'r', encoding='utf-8') as fe:#gbk
                            anno = json.load(fe)
                        obj_list = anno['outputs']['object']
                        img_w = anno['size']['width']
                        img_h = anno['size']['height']
                        img_c = anno['size']['depth']
                        new_str = ''
                        if enable_save_img:
                            img_s = cv2.imread(filename + '.jpg')
                        is_anno_wrong = False
                        for obj in obj_list:
                            # if obj['name'] != folder:
                            if obj['name'] != folder and obj['name'] != 'quqir' and obj['name'] != ' zaocanbao' and obj['name'] != 'quqi ':
                                print("obj['name'] != folder", obj['name'], folder)
                                input()
                            box = obj['bndbox']
                            x_min = box['xmin']
                            x_max = box['xmax']
                            y_min = box['ymin']
                            y_max = box['ymax']
                            if isinstance(x_min, int) and isinstance(y_min, int) and isinstance(x_max, int) and isinstance(y_max, int):
                                print('box check is over.')
                            else:
                                print('box x_min y_min x_max y_max is wrong', x_max, x_max, x_max, x_max)
                                print('press enter to skip this img.')
                                input()
                                is_anno_wrong = True
                                break
                            if (x_min<0 or x_min>=img_w or y_min<0 or y_min>=img_h or 
                                x_max<0 or x_max>=img_w or y_max<0 or y_max>=img_h):
                                need_check = True
                                print('box is error', x_min, y_min, x_max, y_max)
                                box_color = (255, 0, 0)
                                # input()
                            else:
                                box_color = (0, 0, 255)

                            if enable_save_img:
                                cv2.rectangle(img_s, (int(x_min), int(y_min)), (int(x_max), int(y_max)), box_color, 1, 8, 0)
                                bottomLeftCornerOfText = (int(x_min), int(y_min))
                                fontScale = 1.0
                                lineType = 2
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(img_s, folder, bottomLeftCornerOfText, 
                                    font, fontScale, box_color, lineType)

                            if x_min < 0:
                                x_min = 0
                            if y_min < 0:
                                y_min = 0
                            if x_max > img_w-1:
                                x_max = img_w-1
                            if y_max > img_h-1:
                                y_max = img_h-1

                            if (x_min<0 or x_min>=img_w or y_min<0 or y_min>=img_h or 
                                x_max<0 or x_max>=img_w or y_max<0 or y_max>=img_h):
                                print('box is stil error', x_min, y_min, x_max, y_max)
                                input()
                            new_str += " " + ",".join(
                                [str(x_min), str(y_min), str(x_max), str(y_max), str(class_id)]
                                )
                        if enable_save_img:
                            if need_check:
                                cv2.imwrite(str(f_idx) + '-anno-check.jpg', img_s)
                        if is_anno_wrong:
                            print('skip this img because box x_min y_min x_max y_max is wrong.', file_cur)
                            continue
                        if new_str == '':
                            print('new_str=empty, maybe missing box:', file_cur)
                            print('press enter to skip this img.')
                            input()
                            continue
                        image_num += 1
                        image_path = filename + '.jpg'
                        annotation = image_path
                        annotation += new_str
                        annotation += "\n"
                        # print(annotation)
                        f.write(annotation)
                        
                        data_list_f.write(file_tp[:-5] + '\n')
    data_list_f.close()
    return image_num


if __name__ == "__main__":
    last_data_stream_status = '0'
    current_data_stream_status = '1'
    current_note_log = 'custom data writing to list.'
    if not data_stream_status_machine.start_check(last_data_stream_status):
        exit(1)
    start_time = get_time_util.get_last_time()
    print('Start custome data writing to list...')
    
    train_data_path_2007 = '/home/chenp/Yolov5_tf/data/dataset/train'
    train_annotation_path = "/home/chenp/Yolov5_tf/train_annotation.txt"
    if os.path.exists(train_annotation_path):
        print('remove train annotation path...')
        input()
        os.remove(train_annotation_path)
    train_data_list_path = './train.txt'
    
    test_data_path_2007 = '/home/chenp/Yolov5_tf/data/dataset/test'
    test_annotation_path = "/home/chenp/Yolov5_tf/test_annotation.txt"
    if os.path.exists(test_annotation_path):
        print('remove test annotation path...')
        input()
        os.remove(test_annotation_path)
    test_data_list_path = './test.txt'

    len_train = parse_custom_annotation(
        train_data_path_2007,
        train_annotation_path,
        train_data_list_path
    )
    len_test = parse_custom_annotation(
        test_data_path_2007,
        test_annotation_path,
        test_data_list_path
    )

    print(
        "The number of images for train and test are :train : {0} | test : {1}".format(
            len_train, len_test
        )
    )

    print('Custom data writing to list done...')
    end_time = get_time_util.get_last_time()
    data_stream_status_machine.end_check(data_stream_status=current_data_stream_status,
        note_log=current_note_log,
        start_time=start_time, end_time=end_time)