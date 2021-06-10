run success on:
tf version
2.2.0

pipeline:
./train.sh
python3 freeze_ckpt_to_pb.py
python3 convert_tflite.py
