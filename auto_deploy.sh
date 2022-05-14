python3 split_dataset_into_train_test.py 
python3 custom_data.py
./train.sh 
python3 freeze_ckpt_to_pb.py
python3 convert_tflite.py
