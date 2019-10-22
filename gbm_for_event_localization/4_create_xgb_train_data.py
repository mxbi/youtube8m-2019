import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import mlcrate as mlc

meta_valid = mlc.load('validation-framelevel-metadata.pkl')
av_valid = mlc.load('validation-framelevel-avs.pkl')

classes = []
for name, clss, segments in meta_valid:
    for time, cls, y in segments:
        classes.append(cls)

av_preds = mlc.load('./av_preds.pkl')

from sklearn import model_selection
import pickle

clf_xgb_models = {}

for target_cls in tqdm(list(set(classes))):
    x_all = []
    y_all = []
    real_mask = []
    grps = []
    grp_id = 0
    for name, clss, segments in meta_valid:
        grp_id += 1
    #     print(name)
        if target_cls in clss:
            for time, cls, y in segments:
    #             print(segments)
                if cls == target_cls:
                    av_video = av_valid[name]
                    for t in range(time-2, time+7):
                        try:
                            x = av_video[t]
                            x2 = av_preds[name][t]#[target_cls]
#                             print(x.shape, x2.shape)
                            x_all.append(np.append(x, x2))
                            y_all.append(y)
                            real_mask.append(1)
                            grps.append(grp_id)
                        except IndexError as e:
                            print(e)
        else:
            if np.random.random() > 0.91:
                av_video = av_valid[name]
                t = np.random.choice(np.arange(len(av_video)))
                x = av_video[t]
                x2 = av_preds[name][t]#[target_cls]
                x_all.append(np.append(x, x2))
                y_all.append(0)
                real_mask.append(0)
                grps.append(grp_id)
    x_all = np.vstack(x_all)
    y_all = np.array(y_all)
    
    mlc.save([x_all, y_all, grps], './xgb-training/cls_{}.pkl'.format(target_cls))