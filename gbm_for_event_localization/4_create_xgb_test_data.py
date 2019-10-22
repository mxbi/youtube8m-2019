import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import mlcrate as mlc

av_preds = mlc.load('./av_preds.pkl')

sub = pd.read_csv('2018_preds.csv')

all_ids = sub['VideoId'].values

meta_test = mlc.load('test-framelevel-metadata.pkl')
av_test = mlc.load('test-framelevel-avs.pkl')

all_av = []
video_indices = {}
ix = 0
for video in tqdm(all_ids):
    x = av_test[video]
    x2 = av_preds[video]#[target_cls]
    av = np.hstack([x, x2])
    
    video_indices[video] = ix
    all_av.append(av)
    ix += len(av)

all_av = np.concatenate(all_av)

import h5py
h5 = h5py.File('all_av_data.h5', 'w')
h5.create_dataset('all_av', data=all_av, compression='lzf')

h5.close()

mlc.save(video_indices, 'all_av_video_indices.pkl')