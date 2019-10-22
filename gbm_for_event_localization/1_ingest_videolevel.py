import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import mlcrate as mlc
import tensorflow as tf
import gc
import h5py

train_records = sorted(glob('yt8m-video/train*.tfrecord'))
val_records = sorted(glob('yt8m-video/validate*.tfrecord'))
test_records = sorted(glob('yt8m-video/test*.tfrecord'))

def process_record(record):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with sess.as_default():
            metadata = []

            for example in tf.python_io.tf_record_iterator(record):
                tf_example = tf.train.Example.FromString(example)
                vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
                vid_labels = list(tf_example.features.feature['labels'].int64_list.value)
                mean_video = tf_example.features.feature['mean_rgb'].float_list.value
                mean_audio = tf_example.features.feature['mean_audio'].float_list.value
                metadata.append((vid_id, vid_labels, np.concatenate([mean_video, mean_audio])))

            return metadata

test_metadata = []
for record in tqdm(test_records):
    test_metadata.extend(process_record(record))

for i in tqdm(range(len(test_metadata))):
    test_metadata[i] = (test_metadata[i][0], list(test_metadata[i][1]), test_metadata[i][2])

h5_test = h5py.File('test_videolevel.h5', 'w')
h5_test.create_dataset('data', data=np.vstack([x[2] for x in test_metadata]), compression='lzf', chunks=(16384, 1152))
h5_test.close()

mlc.save([x[:2] for x in test_metadata], 'test_videolevel_meta.pkl')

gc.collect()

train_metadata = []
for record in tqdm(train_records):
    train_metadata.extend(process_record(record))
    
h5 = h5py.File('train_videolevel.h5', 'w')
h5.create_dataset('data', data=np.vstack([x[2] for x in train_metadata]), compression='lzf', chunks=(16384, 1152))
mlc.save([x[:2] for x in train_metadata], 'train_videolevel_meta.pkl')
h5.close()

mlc.save(train_metadata, 'train_videolevel.pkl')

val_metadata = []
for record in tqdm(val_records):
    val_metadata.extend(process_record(record))
    
h5 = h5py.File('val_videolevel.h5', 'w')
h5.create_dataset('data', data=np.vstack([x[2] for x in val_metadata]), compression='lzf', chunks=(16384, 1152))
mlc.save([x[:2] for x in val_metadata], 'val_videolevel_meta.pkl')
h5.close()

mlc.save(val_metadata, 'val_videolevel.pkl')