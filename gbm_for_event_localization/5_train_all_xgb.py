import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import mlcrate as mlc
import subprocess
import multiprocessing
from tqdm import tqdm

meta_valid = mlc.load('validation-framelevel-metadata.pkl')

classes = []
for name, clss, segments in meta_valid:
    for time, cls, y in segments:
        classes.append(cls)

classes = list(set(classes))

print('{} Classes: {}'.format(len(classes), classes))

def run_cls(target_cls):
    print('Starting class {}'.format(target_cls))
    t0 = mlc.time.Timer()

    output = str(subprocess.check_output(['python', 'train_cls_xgb.py', str(target_cls)]))

    print('Finished class {}, time {}: {}'.format(target_cls, t0.fsince(), output.split('\\n')[-2]))

pool = multiprocessing.Pool(8)

for _ in tqdm(pool.imap_unordered(run_cls, classes), total=len(classes)):
    pass
