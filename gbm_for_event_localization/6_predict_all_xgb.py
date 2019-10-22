import pandas as pd
import numpy as np
import mlcrate as mlc
import h5py
from glob import glob
import xgboost as xgb
from tqdm import tqdm
import gc

vocab = pd.read_csv('vocabulary.csv')
classes = set(vocab.Index.values)
class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

all_indices = mlc.load('all_av_video_indices.pkl')

h5 = h5py.File('all_av_data.h5')['all_av']
print(h5.shape)
size = h5.shape[0]
chunksize = size // 5

print('loading models')
models = [mlc.load(x) for x in glob('./xgb-models/*.pkl')]
print('loaded')

class CLSEnsemble:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, darr):
        # darr = xgb.DMatrix(arr)
        preds = []
        for model in self.models:
            preds.append(model.predict(darr))
        corr = np.corrcoef(preds)
        print('corr', corr[0, 1])
        preds = np.mean(preds, axis=0)
        return preds

cls_modelsx = {}
for cls, m in models:
    cls_modelsx[cls] = CLSEnsemble(m)

all_preds = np.zeros((size, 1000), dtype=np.float16)

for chunkid in range(5):
    print('chunk', chunkid)
    start = chunkid*chunksize
    end = None if chunkid == 4 else (chunkid+1)*chunksize
    av = h5[start:end]
    print(av.shape)
    darr = xgb.DMatrix(av)

    for cls in tqdm(classes):
        all_preds[start:end, class_to_idx[cls]] = cls_modelsx[cls].predict_proba(darr)

    del av, darr
    gc.collect()

print('done! saving')
mlc.save(all_preds, 'xgb4_pred_matrix.pkl')
