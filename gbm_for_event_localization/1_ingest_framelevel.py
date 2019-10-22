import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import mlcrate as mlc

import tensorflow as tf

sess = tf.InteractiveSession()

def process_record(record):
    tf.reset_default_graph()
    with tf.Session() as sess:
        with sess.as_default():
            metadata = []
            all_av = {}

            for example in tf.python_io.tf_record_iterator(record):
                tf_example = tf.train.Example.FromString(example)
                vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
                vid_labels = tf_example.features.feature['labels'].int64_list.value
                segment_start_times  = tf_example.features.feature['segment_start_times'].int64_list.value
            #     segment_end_times = tf_example.features.feature['segment_end_times'].int64_list.value
                segment_labels = tf_example.features.feature['segment_labels'].int64_list.value
                segment_scores = tf_example.features.feature['segment_scores'].float_list.value
                metadata.append((vid_id, list(vid_labels), list(zip(segment_start_times, segment_labels, segment_scores))))

                x, y = tf.parse_single_sequence_example(example, sequence_features={'rgb': tf.FixedLenSequenceFeature([], dtype=tf.string), 'audio': tf.FixedLenSequenceFeature([], dtype=tf.string)})
                rgb_frames = tf.decode_raw(y['rgb'], tf.uint8).eval()
                audio_frames = tf.decode_raw(y['audio'], tf.uint8).eval()
                av = np.concatenate([rgb_frames, audio_frames], 1)

                all_av[vid_id] = av

            return metadata, all_av

val_records = sorted(glob('yt8m-frame/validate*.tfrecord'))

metadatas, all_avs = [], {}
for record in tqdm(val_records):
    metadata, all_av = process_record(record)
    metadatas.extend(metadata)

mlc.save(all_avs, 'validation-framelevel-avs.pkl')

for i in tqdm(range(len(metadatas))):
    metadatas[i] = (metadatas[i][0], list(metadatas[i][1]), metadatas[i][2])

mlc.save(metadatas, 'validation-framelevel-metadata.pkl')

test_records = sorted(glob('yt8m-frame/test*.tfrecord'))

metadatas, all_avs = [], {}
for record in tqdm(test_records):
    metadata, all_av = process_record(record)
    metadatas.extend(metadata)
    all_avs.update(all_av)

mlc.save(all_avs, 'test-framelevel-avs.pkl')
mlc.save(metadatas, 'test-framelevel-metadata.pkl')