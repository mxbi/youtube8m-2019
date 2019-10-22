import numpy as np
import pandas as pd
import mlcrate as mlc
from tqdm import tqdm
from collections import defaultdict
import gzip

vocab = pd.read_csv('vocabulary.csv')
classes = set(vocab.Index.values)
class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

sub = pd.read_csv('youtube8mchallenge/preds.csv')

pred_pairs = defaultdict(list)

for videoid, string in sub.values:
    preds = string.split(' ')
    i = 0
    while i < len(preds):
        lab = int(preds[i])
        conf = preds[i+1]
        pred_pairs[lab].append((videoid, float(conf)))
        i += 2

print('Loading prediction matrix...')
xgb_pred_matrix = mlc.load('xgb4_pred_matrix.pkl')
video_indices = mlc.load('all_av_video_indices.pkl')
print('Done!')

out = gzip.open('yt8m2_xgb4_pow4.csv.gz','wt')
out.write('Class,Segments\n')

# Power parameter for model composition
POW = 4

def process_class(cls):
    pairs = sorted(pred_pairs[cls], reverse=True, key=lambda x: x[1])
    thresh = 2e-4
    videos = [p[0] for p in pairs if p[1] > thresh]
    pairs = [p for p in pairs if p[1] > thresh]
    assert len(videos) == len(pairs)

    cls_model_preds = xgb_pred_matrix[:, class_to_idx[cls]]

    preds = []
    for _id, conf in pairs:
        for i in range(60):
#                 print(video_indices[_id] + i*5, video_indices[_id]+(i+1)*5)
            probs = cls_model_preds[video_indices[_id]+i*5-2:video_indices[_id]+(i+1)*5+2] ** POW
            p = probs.mean() if len(probs) > 0 else 0
#                 print(conf, p)
            preds.append(('{}:{}'.format(_id, i*5), p*conf))
    print(len(preds))
    sorted_preds = sorted(preds, reverse=True, key=lambda x:x[1])

    return (cls, '{},{}\n'.format(cls, ' '.join([p[0] for p in sorted_preds])))

from multiprocessing import Pool
pool = Pool(16)

results = {}

for cls, preds in tqdm(pool.imap_unordered(process_class, classes), total=1000):
    results[cls] = preds

for cls in tqdm(classes):
    out.write(results[cls])

out.close()
