# GBM Solution

Before running this solution, you need to setup the Youtube-8M input data:
- The frame-level `.tfrecord` files for Youtube-8M (v3), validation and test sets only, need to go in `./yt8m-frame`
- The video-level `.tfrecord` files for Youtube-8M Segments (v3), train, validate and test, need to go in `./yt8m-valid`
- `vocabulary.csv` needs to exist in this directory - this is already provided in the repo.

### 2018 Pre-trained model video-level predictions

First, the pre-trained 2018 [Next Top GB Model](https://github.com/miha-skalic/youtube8mchallenge) model needs to be run on the test set, to generate the top 500 classes with probabilities for each video.

```bash
git clone https://github.com/miha-skalic/youtube8mchallenge
# Download pre-trained model from https://drive.google.com/open?id=1hrHOWc_3xFk1WofTnimq8icjzJ-k9pnh here
cd youtube8mchallenge
python inference_gpu.py --input_data_pattern "../yt8m-frame/test*" --input_model_tgz model.tgz --output_file ../2018_preds.csv --top_k 500 --batch_size 32
cd ..
```

Essentially, the top 500 predictions for each video should be saved in a file `2018_preds.csv`, in this directory before proceeding.

## Solution

Run `./run_solution.sh` to run the entire solution from start to finish (can take a couple days!)

```bash
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
```

This will create a submission file `yt8m2_xgb4_pow4.csv.gz` (LARGE!), which when uploaded to Kaggle should score ~0.795 on the private leaderboard.