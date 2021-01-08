import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2
import core.utils as utils
import os
from core.config import cfg

# flags.DEFINE_string('weights', './checkpoint/social_yolov3_test-loss=3.3218.ckpt-51.pb', 'path to weights file')
# flags.DEFINE_string('weights', './checkpoint-v2/qixing_yolov3_test-loss=1.5639.ckpt-567.pb', 'path to weights file')
#flags.DEFINE_string('weights', './checkpoint-v2-2020-12-31_13-26-59/qixing_yolov3_test-loss=6.7334.ckpt-65.pb', 'path to weights file')
# flags.DEFINE_string('weights', './checkpoint-v2-2020-12-31_21-01-16/qixing_yolov3_test-loss=5.7523.ckpt-842.pb', 'path to weights file')
#flags.DEFINE_string('weights', './checkpoint-v2-2021-01-04_17-52-45/qixing_yolov3_test-loss=3.1826.ckpt-28.pb', 'path to weights file')
flags.DEFINE_string('weights', './checkpoint-v2-2021-01-04_17-52-45/qixing_yolov3_test-loss=1.8997.ckpt-409.pb', 'path to weights file')
#flags.DEFINE_string('weights', './checkpoint-v2-2021-01-04_17-52-45/qixing_yolov3_test-loss=1.5758.ckpt-669.pb', 'path to weights file')

flags.DEFINE_string('output', './1.8997-409-detecor.tflite', 'path to output')
flags.DEFINE_string('output_cplus', './1.8997-409-detecor-cplus.tflite', 'path to output')
flags.DEFINE_integer('input_size', 416, 'path to output')
flags.DEFINE_string('quantize_mode', 'float32', 'quantize mode (int8, float16, float32)')
flags.DEFINE_string('dataset', "/Volumes/Elements/imgs/coco_dataset/coco/5k.txt", 'path to dataset')

def representative_data_gen():
  fimage = open(FLAGS.dataset).read().split()
  for input_value in range(10):
    if os.path.exists(fimage[input_value]):
      original_image=cv2.imread(fimage[input_value])
      original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
      image_data = utils.image_preprocess(np.copy(original_image), [FLAGS.input_size, FLAGS.input_size])
      img_in = image_data[np.newaxis, ...].astype(np.float32)
      print("calibration image {}".format(fimage[input_value]))
      yield [img_in]
    else:
      continue

def save_tflite(input_arrays, output_arrays, out_tflite_model_name):
  # converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.weights)
  # converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
  #         graph_def_file, input_arrays, output_arrays)
  converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        FLAGS.weights, input_arrays, output_arrays, {input_arrays[0] :[1,FLAGS.input_size,FLAGS.input_size,3]})
  if FLAGS.quantize_mode == 'float16':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
  elif FLAGS.quantize_mode == 'int8':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.representative_dataset = representative_data_gen

  tflite_model = converter.convert()
  open(out_tflite_model_name, 'wb').write(tflite_model)
  logging.info("model saved to: {}".format(out_tflite_model_name))

def demo():
  img_path_file = '/home/chenp/YOLOv4-pytorch/qixing-data/test' #argv[3]
  # img_path_file = '/home/chenp/YOLOv4-pytorch/qixing-data/test/zhibeidangao/test-z' #argv[3]
  # out_path = 'det_out-tflite-1.5758.ckpt-669' #argv[4]
  out_path = 'det_out-tflite-1.8997.ckpt-409-only-middle' #argv[4]
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  if not os.path.exists(img_path_file):
      print('img_path_file=%s not exist' % img_path_file)
      sys.exit()
  interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
  interpreter.allocate_tensors()
  logging.info('tflite model loaded')

  input_details = interpreter.get_input_details()
  print(input_details)
  output_details = interpreter.get_output_details()
  print(output_details)

  input_shape = input_details[0]['shape']
  img_files = []
  for files in os.walk(img_path_file):
    print(files[0])
    for tp_f in files[2]:
      img_files.append(files[0] + '/' + tp_f)
    # input()
  for idx, in_img_file in enumerate(img_files):
    #print('idx=', idx, 'in_img_file=', in_img_file)
    if not os.path.exists(in_img_file):
      print('idx=', idx, 'in_img_file=', in_img_file, ' not exist')
      continue

    img = cv2.imread(in_img_file)
    if in_img_file[-4:] != '.jpg':
      print('not jpg ', in_img_file)
      continue
    if img is None:
      print('idx=', idx, 'in_img_file=', in_img_file, ' read error')
      continue
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    img_size = img.shape[:2]
    image_data = utils.image_preporcess(np.copy(img), [FLAGS.input_size, FLAGS.input_size])
    input_data = image_data[np.newaxis, ...].astype(np.float32)

    # input_data0 = np.ones((1, 416, 416), np.float32)*0.2
    # input_data1 = np.ones((1, 416, 416), np.float32)*0.6
    # input_data2 = np.ones((1, 416, 416), np.float32)*0.5
    # input_data = np.stack((input_data0, input_data1, input_data2), axis=3)
    print('input_data', input_data.shape, input_data.dtype)
    print(input_data[0, 0, 0, 0], input_data[0, 0, 0, 1], input_data[0, 0, 0, 2],
    input_data[0, 200, 200, 0], input_data[0, 200, 200, 1], input_data[0, 200, 200, 2])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    print('run over.')
    # input()
    print(type(output_data))
    pred_sbbox, pred_mbbox, pred_lbbox = output_data
    print(pred_sbbox.shape, pred_mbbox.shape, pred_lbbox.shape, img_size)
    
    #print(pred_sbbox[0, 20, 20, 0, 4], 
    # pred_sbbox[0, 20, 20, 0, 5], pred_sbbox[0, 20, 20, 0, 6], pred_sbbox[0, 20, 20, 0, 7], pred_sbbox[0, 20, 20, 0, 8])
    #print(pred_sbbox[0, 0, 0, 0, 0], pred_sbbox[0, 0, 0, 0, 1], pred_sbbox[0, 0, 0, 0, 2], pred_sbbox[0, 0, 0, 0, 3], pred_sbbox[0, 0, 0, 0, 4], 
    # pred_sbbox[0, 0, 0, 0, 5], pred_sbbox[0, 0, 0, 0, 6], pred_sbbox[0, 0, 0, 0, 7], pred_sbbox[0, 0, 0, 0, 8])
    print(in_img_file)
    num_classes = 4
    score_thresh = 0.6
    iou_type = 'iou' #yolov4:diou, else giou
    iou_thresh = 0.3
    
    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                              np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                              np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
    '''
    pred_bbox = np.concatenate([
                                np.reshape(pred_mbbox, (-1, 5 + num_classes))
                                ], axis=0)
    '''
    bboxes = utils.postprocess_boxes(pred_bbox, img_size, FLAGS.input_size, score_thresh)
    bboxes = utils.nms(bboxes, iou_type, iou_thresh, method='nms')
    score = 0
    image = utils.draw_bbox(img, bboxes)
    #image = Image.fromarray(image)
    #image.show()
    if len(bboxes) > 0:
      score = bboxes[0][4]
      print('bboxes len(bboxes) > 0', type(bboxes))
    else:
      print('bboxes len(bboxes) = 0', type(bboxes))
      score = 0
    out_img = np.asarray(image)

    file_path, file_name = os.path.split(in_img_file)
    file, postfix = os.path.splitext(file_name)
    out_file = os.path.join(out_path, file + '_%.6f' % (score) + postfix)

    cv2.imwrite(out_file, out_img)
    print('idx=', idx, 'in_img_file=', in_img_file, 'out_file=', out_file)

def main(_argv):
  input_arrays = ['input/input_data']
  output_arrays = ['pred_sbbox/concat_2', 'pred_mbbox/concat_2', 'pred_lbbox/concat_2']
  save_tflite(input_arrays, output_arrays, FLAGS.output)
  print('convert done...')
  input_arrays = ['input/input_data']
  # output_arrays_for_CPlus = ['pred_lbbox/Exp', 'pred_lbbox/Sigmoid', 'pred_lbbox/Sigmoid_1', 'pred_lbbox/Sigmoid_2', 
  #  'pred_mbbox/Exp', 'pred_mbbox/Sigmoid', 'pred_mbbox/Sigmoid_1', 'pred_mbbox/Sigmoid_2',
  #  'pred_sbbox/Exp', 'pred_sbbox/Sigmoid', 'pred_sbbox/Sigmoid_1', 'pred_sbbox/Sigmoid_2']
  output_arrays_for_CPlus = ["conv_sbbox/BiasAdd", "conv_mbbox/BiasAdd", "conv_lbbox/BiasAdd"]
  save_tflite(input_arrays, output_arrays_for_CPlus, FLAGS.output_cplus)

  demo()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
