tflite_convert \
  --output_file=./detector.tflite \
  --graph_def_file=./checkpoint-v2/social_yolov3_test-loss=56.9129.ckpt-2.pb \
  --input_arrays=input/input_data \
  --output_arrays=pred_sbbox/concat_2,pred_mbbox/concat_2,pred_lbbox/concat_2 \
  --input_shape=1,416,416,3 \
  --enable_v1_converter