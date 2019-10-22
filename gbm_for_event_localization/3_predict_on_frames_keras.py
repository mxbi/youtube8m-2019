import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import mlcrate as mlc

import keras as k
import keras.layers as l

# train_meta = mlc.load('train_videolevel_meta.pkl')
# val_meta = mlc.load('val_videolevel_meta.pkl')
# test_meta = mlc.load('test_videolevel_meta.pkl')

# import h5py
# h5_train = h5py.File('train_videolevel.h5', 'r')['data']
# h5_val = h5py.File('val_videolevel.h5', 'r')['data']
# h5_test = h5py.File('test_videolevel.h5', 'r')['data']

av_valid = mlc.load('validation-framelevel-avs.pkl')

mdl = k.models.load_model('./basic_model_15epoch.keras')

av_preds = {}

for _id, video in tqdm(av_valid.items()):
    p = mdl.predict_on_batch(video * (4/255) - 1.9921875)
    av_preds[_id] = p.astype(np.float16)

av_test = mlc.load('test-framelevel-avs.pkl')

for _id, video in tqdm(av_test.items()):
    p = mdl.predict_on_batch(video * (4/255) - 1.9921875)
    av_preds[_id] = p.astype(np.float16)

mlc.save(av_preds, 'av_preds.pkl')