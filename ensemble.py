import pandas as pd
import sys
import numpy as np
import mlcrate as mlc
from tqdm import tqdm

assert len(sys.argv) == 3, "Usage: {} [submission1.csv[.gz]] [submission2.csv[.gz]] [output_submission.csv[.gz]]".format(sys.argv[0])

print('Reading submission files...')
x1 = pd.read_csv(sys.argv[0])
x2 = pd.read_csv(sys.argv[1])

top1 = [x.split(' ')[0] for x in x1['Segments'].values]
top2 = [x.split(' ')[0] for x in x2['Segments'].values]

print('Proportion of classes with same top1: ', (np.array(top1) == np.array(top2)).mean())

vid1 = [x.split(':')[0] for x in x1['Segments'].values]
vid2 = [x.split(':')[0] for x in x2['Segments'].values]

print('Computing ensemble...')

WEIGHTS = [1, 1]

preds = []
for a, b in zip(tqdm(x1['Segments'].values), x2['Segments'].values):
    c = np.array(a.split(' '))
    d = np.array(b.split(' '))
    print(len(c), len(d))

    out = mlc.ensemble.rank_average(c, d, weights=WEIGHTS)
    
    print(c[:10], d[:10])
    print(out[:10])
    preds.append(' '.join(out))

print('Saving submission...')

x3=x1.copy()
x3['Segments'] = preds

mlc.kaggle.save_sub(x3, sys.argv[3])
