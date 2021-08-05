import shutil
import os
import glob
import random
import yolov4_config as cfg

# train_dst_folder='/home/chenp/YOLOv4-pytorch/qixing-data/train'
# test_dst_folder='/home/chenp/YOLOv4-pytorch/qixing-data/test'
# new_dataset_folder='/home/chenp/YOLOv4-pytorch/qixing1214'

# train_dst_folder='/home/chenp/Yolov5_tf/data/dataset/train'
# test_dst_folder='/home/chenp/Yolov5_tf/data/dataset/test'
# new_dataset_folder='/home/chenp/Yolov5_tf/data/dataset/data-0124'

train_dst_folder='/home/chenp/Yolov5_tf/data/dataset/train'
test_dst_folder='/home/chenp/Yolov5_tf/data/dataset/test'
new_dataset_folder='/home/chenp/Yolov5_tf/data/dataset/data-0406'

bad_data_dst_folder='/data/bad_data'

def add_new_data(new_dataset_folder,
    train_dst_folder, test_dst_folder,
    test_set_proportion=0.2):
    # class_name_list = ['zhibeidangao', 'qifeng', 'tusi', 'quqi', 'zaocanbao', 'dangaojuan',
    #  'danta', 'jichi', 'jichigen', 'jixiongrou', 'jimihua', 'manyuemeibinggan',
    #  'peigen', 'niupai', 'shutiao', 'oubao']
    # print('Please Confirm class_name_list: ', class_name_list)
    # input()

    class_name_list = cfg.Customer_DATA["CLASSES"]
    print('Please Confirm class_name_list: ', class_name_list)
    input()
    
    add_folder_num = 0
    add_test_data_num = 0
    add_train_data_num = 0

    if not os.path.exists(test_dst_folder):
        print('test_dst_folder does not exist: ', test_dst_folder)
        input()
        return False, add_train_data_num, add_test_data_num
    if not os.path.exists(train_dst_folder):
        print('train_dst_folder does not exist: ', train_dst_folder)
        input()
        return False, add_train_data_num, add_test_data_num
    
    if not os.path.exists(bad_data_dst_folder):
        print('bad_data_dst_folder does not exist: ', bad_data_dst_folder)
        os.makedirs(bad_data_dst_folder)
    
    data_folder_list = os.listdir(new_dataset_folder)
    for root_sub_file in data_folder_list:
        folder_path = new_dataset_folder + '/' + root_sub_file
        if not os.path.isdir(folder_path):
            continue

        print('*****************************************move target folder: ', folder_path)
        class_name = root_sub_file.split('-')[0]
        if class_name not in class_name_list:
            print('class_name not in class_name_list: ', class_name)
            input()
            
        data_path_list = []
        bad_data_path_list = []
        file_list = os.listdir(folder_path)
        for file_idx, file_path in enumerate(file_list):
            filename, file_type = os.path.splitext(folder_path + '/' + file_path)
            if file_type == '.json':
                if os.path.exists(filename + '.jpg'):
                    img_path = filename + '.jpg'
                    data_path_list.append(
                        {folder_path + '/' + file_path : img_path}
                        )
                else:
                    bad_data_path_list.append(folder_path + '/' + file_path)
            elif file_type != '.json' and file_type != '.jpg':
                bad_data_path_list.append(folder_path + '/' + file_path)
            elif file_type == '.jpg':
                if not os.path.exists(filename + '.json'):
                    bad_data_path_list.append(folder_path + '/' + file_path)
        for bad_data in bad_data_path_list:
            print('Move bad data: ', bad_data, ' To: ', bad_data_dst_folder)
            shutil.move(bad_data, bad_data_dst_folder)
        
        test_set_size = int(test_set_proportion * len(data_path_list) + 0.5)
        testdata_path_list = random.sample(data_path_list, test_set_size)
        print(testdata_path_list)
        print('test data num: ', len(testdata_path_list), test_set_size, len(data_path_list))
        print('press enter to continue...')
        input()
        dst_class_folder = test_dst_folder + '/' + class_name
        if not os.path.exists(dst_class_folder):
            print('dst_class_folder not exist: ', dst_class_folder)
            print('press enter to mkdir it. ')
            input()
            os.mkdir(dst_class_folder)
        sub_dst_folder = dst_class_folder + '/' + root_sub_file
        if not os.path.exists(sub_dst_folder):
            os.mkdir(sub_dst_folder)

        #move to test_dst_folder:
        for test_data in testdata_path_list:
            json_path = list(test_data.keys())[0]
            print('Move: ', json_path, ' To: ', sub_dst_folder)
            shutil.move(json_path, sub_dst_folder)
            print('Move: ', test_data[json_path], ' To: ', sub_dst_folder)
            shutil.move(test_data[json_path], sub_dst_folder)

        add_test_data_num += len(testdata_path_list)
        add_train_data_num += (len(data_path_list) - len(testdata_path_list))

        dst_class_folder = train_dst_folder + '/' + class_name
        if not os.path.exists(dst_class_folder):
            print('dst_class_folder not exist: ', dst_class_folder)
            print('press enter to mkdir it. ')
            input()
            os.mkdir(dst_class_folder)

        #move to train_dst_folder:
        print('Move: ', folder_path, ' To: ', dst_class_folder)
        shutil.move(folder_path, dst_class_folder)

        add_folder_num += 1
    print('Totally move ', add_folder_num, ' new folders, ', add_train_data_num + add_test_data_num, ' imgs.\n')
    print('Add_train_data_num: ', add_train_data_num)
    print('Add_test_data_num: ', add_test_data_num)
    return True, add_train_data_num, add_test_data_num


if add_new_data(new_dataset_folder=new_dataset_folder,
    train_dst_folder=train_dst_folder, test_dst_folder=test_dst_folder,
    test_set_proportion=0.2):
    print('add success...')
else:
    print('add failed...')
    input()
