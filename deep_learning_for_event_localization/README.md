# Deep learning for event localization

Executing commands listed bellow will train models and make predictions for test examples. Please adjust paths according to your setup.  
`requirements.yaml` file list dependencies needed to run the code.

After prediction files for the models are created you can use `combine.ipynb` to generate a submission file. Adjust file paths at the beginning accordingly.  

## Localization models
```
# Train embedding localization
python train_localization/train_torched_embedd.py \
       --train_data_pattern="./frame_validate_seq/*.tfrecord" \
       --train_dir=embedd_training --batch_size=512 \
       --num_epochs=20 --mask_unlabeled

# Do the inference
for X in {0..999}; do
    python inference_torched_embedd.py \
       --batch_size 100 --input_data_pattern=./frame_test_seq/*.tfrecord \
       --train_dir=/tmp/tmp \
       --segment_label_ids_file ./youtube-8m/segment_label_ids.csv \
       --prediction_file="/torched_embedd/predfile" --embedd ${X} --device 1
done
```

## 5 frame models

### VLAD models


```
# Train the ininital model
python train.py \
  --train_data_pattern=./frame_train/*.tfrecord \
  --model=NetVLADModelLF \
  --start_new_model \
  --train_dir="model/VLAD_5run2" \
  --frame_features=True \
  --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --batch_size=1024 \
  --base_learning_rate=0.001 \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 \
  --iterations=5 \
  --sample_random_frames \
  --subsample \
  --learning_rate_decay=0.8 \
  --netvlad_relu=False \
  --lightvlad=False \
  --gating=True \
  --moe_prob_gating=True \
  --num_gpu 1 \
  --num_epochs=20

# Finetune for localization
python youtube-8m/train.py \
  --train_data_pattern="./frame_validate_seq/*.tfrecord" \
  --model=NetVLADModelLF \
  --start_new_model \
  --train_dir="./VLAD_5frames_run2_finetune" \
  --frame_features=True \
  --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --batch_size=1024 \
  --base_learning_rate=0.0005 \
  --netvlad_cluster_size=256 \
  --netvlad_hidden_size=1024 \
  --moe_l2=1e-6 \
  --iterations=5 \
  --sample_random_frames \
  --learning_rate_decay=0.8 \
  --netvlad_relu=False \
  --lightvlad=False \
  --gating=True \
  --moe_prob_gating=True \
  --num_gpu 1 \
  --num_epochs=3 \
  --segment_labels \
  --mask \
  --seed_weights ./model.ckpt-50842

# Generate predictions
python inference_direct.py \
   --model=NetVLADModelLF \
   --frame_features=True \
   --feature_names="rgb,audio" \
   --feature_sizes="1024,128" \
   --netvlad_cluster_size=256 \
   --netvlad_hidden_size=1024 \
   --moe_l2=1e-6 \
   --iterations=5 \
   --netvlad_relu=False \
   --lightvlad=False \
   --gating=True \
   --moe_prob_gating=True \
   --segment_labels \
   --batch_size 10 \
   --input_data_pattern=frame_test_seq/*.tfrecord \
   --train_dir=VLAD_5frames_run2_finetune  \
   --output_file="VLAD5frames_50842_686_prediction.csv" \
   --submission_file "/tmp/gone.csv" \
   --segment_label_ids_file segment_label_ids.csv \
   --checkpoint 686 \
   --prediction_file="shared/VLAD5frames_50842_686"
```

### DBoF model

```
# Train the ininital model
python train.py \
   --frame_features \
   --train_data_pattern="frame_train/*.tfrecord" \
   --feature_names=rgb,audio \
   --feature_sizes=1024,128 \
   --train_dir=./dbof_trainandval \
   --start_new_model \
   --model=DbofModel \
   --iterations=5 \
   --epochs=20 \
   --num_gpu=1 \
   --batch_size=2048 \
   --subsample

# Finetune for localization
python train.py \
   --base_learning_rate=0.002 \
   --frame_features \
   --train_data_pattern="./*.tfrecord" --feature_names=rgb,audio \
   --feature_sizes=1024,128 --train_dir=./DBoF_finetune_wmask --start_new_model \
   --model=DbofModel --iterations=5 --num_gpu=1 --batch_size=2048 \
   --segment_labels --mask \
   --seed_weights /mnt/4tbyoutube/model/dbof_trainandval/model.ckpt-11637
   
# Generate predictions
python inference_direct.py \
   --model=DbofModel \
   --batch_size 100 \
   --input_data_pattern=frame_test_seq/*.tfrecord \
   --train_dir=./DBoF_finetune_wmask \
   --output_file="./DBoF_finetune_wmask/DBoF_finetune_wmask_cp572.csv" \
   --submission_file "/tmp/gone.csv" \
   --segment_label_ids_file segment_label_ids.csv \
   --checkpoint 572 \
   --prediction_file="./dbof_finetune_wmask_cp752"
```

## Global model

Please use inference from [2nd year model](https://github.com/miha-skalic/youtube8mchallenge) to generate predictions.
 
