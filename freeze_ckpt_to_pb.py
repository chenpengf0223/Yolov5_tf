# -*- coding: utf-8 -*-
import os
import sys
from core.yolov3 import YOLOV3
from core.yolov4 import YOLOV4
from core.yolov5 import YOLOV5
import get_time_util
import data_stream_status_machine
import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


if __name__ == "__main__":
    last_data_stream_status = '2'
    current_data_stream_status = '3'
    current_note_log = 'freeze ckpt to pb.'
    if not data_stream_status_machine.start_check(last_data_stream_status):
        exit(1)
    start_time = get_time_util.get_last_time()
    print('Start freezing ckpt to pb...')

    """
    argv = sys.argv
    if len(argv) < 5:
        print('usage: python freeze_ckpt_to_pb.py gpu_id net_type(yolov5/yolov4/yolov3) ckpt_file pb_file')
        sys.exit()
    """
    gpu_id = '0' #argv[1]
    net_type = 'yolov3' #argv[2]
    # ckpt_file = 'checkpoint-v2-2020-12-31_13-26-59/qixing_yolov3_test-loss=6.7334.ckpt-65' #
    # ckpt_file = 'checkpoint-v2/qixing_yolov3_test-loss=1.5639.ckpt-567' #argv[3]
    # ckpt_file = 'checkpoint-v2-2021-01-04_17-52-45/qixing_yolov3_test-loss=3.1826.ckpt-28'
    # ckpt_file = 'checkpoint-v2-2020-12-31_21-01-16/qixing_yolov3_test-loss=5.7523.ckpt-842'

    # ckpt_file = 'checkpoint-v2-2021-01-04_17-52-45/qixing_yolov3_test-loss=1.8997.ckpt-409'
    #ckpt_file = 'checkpoint-v2-2021-01-04_17-52-45/qixing_yolov3_test-loss=1.5758.ckpt-669'
    #ckpt_file = 'checkpoint-v2-2021-01-28_16-02-28/qixing_yolov3_test-loss=3.5855.ckpt-382'
    #ckpt_file = 'checkpoint-v2-2021-04-08_14-14-48/qixing_yolov3_test-loss=4.9424.ckpt-653'
    f_in = open('./best-ckpt-path', 'r')
    ckpt_file = f_in.readline()
    f_in.close()

    ckpt_file = ckpt_file.strip()
    print('ckpt_file confirm: ', ckpt_file)
    input()
    if not os.path.exists(ckpt_file + '.index'):
        print('freeze_ckpt_to_pb ckpt_file=', ckpt_file, ' not exist')
        sys.exit()

    pb_file = ckpt_file + '.pb' #argv[4]

    #set config for weight pb path:
    out_f = open('./weight-pb-path', 'w')
    out_f.write(pb_file)
    out_f.close() 
    print('freeze_ckpt_to_pb gpu_id=%s, net_type=%s, ckpt_file=%s, pb_file=%s' % (gpu_id, net_type, ckpt_file, pb_file))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]
    #output_node_names = ["input/input_data", "conv_sbbox/BiasAdd", "conv_mbbox/BiasAdd", "conv_lbbox/BiasAdd"]
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')

    if net_type == 'yolov3':
        model = YOLOV3(input_data, trainable=False, freeze_pb=False)
    elif net_type == 'yolov4':
        model = YOLOV4(input_data, trainable=False)
    elif net_type == 'yolov5':
        model = YOLOV5(input_data, trainable=False)
    else:
        print('net_type=', net_type, ' error')

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def=sess.graph.as_graph_def(),
                                                                       output_node_names=output_node_names)
    with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(converted_graph_def.SerializeToString())

    
    print('Freezing ckpt to pb done...')
    end_time = get_time_util.get_last_time()
    data_stream_status_machine.end_check(data_stream_status=current_data_stream_status,
        note_log=current_note_log,
        start_time=start_time, end_time=end_time)
