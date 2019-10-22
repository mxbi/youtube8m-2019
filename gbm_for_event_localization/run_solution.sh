#!/bin/bash

mkdir xgb-training xgb-models

set -e

python 1_ingest_framelevel.py
python 1_ingest_videolevel.py
python 2_train_videolevel_keras.py
python 3_predict_on_frames_keras.py
python 4_create_xgb_train_data.py
python 4_create_xgb_test_data.py
python 5_train_all_xgb.py
python 6_predict_all_xgb.py
python 7_create_submission.py