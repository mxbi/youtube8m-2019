import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import mlcrate as mlc

import keras as k
import keras.layers as l

train_meta = mlc.load('train_videolevel_meta.pkl')
val_meta = mlc.load('val_videolevel_meta.pkl')
test_meta = mlc.load('test_videolevel_meta.pkl')

import h5py
h5_train = h5py.File('train_videolevel.h5', 'r')['data']
h5_val = h5py.File('val_videolevel.h5', 'r')['data']
h5_test = h5py.File('test_videolevel.h5', 'r')['data']

vocab = pd.read_csv('vocabulary.csv')
classes = set(vocab.Index.values)
class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

class YT8MSequence(k.utils.Sequence):

    def __init__(self, h5, meta, batch_size):
        self.h5 = h5
        self.meta = meta
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.h5) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.h5[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_meta = self.meta[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = np.zeros((len(batch_x), 1000))
        for i, (_, clss) in enumerate(batch_meta):
            for cls in clss:
                if cls in class_to_idx:
                    batch_y[i, class_to_idx[cls]] = 1

        return batch_x, batch_y

train_seq = YT8MSequence(h5_train, train_meta, 16384)

val_seq = YT8MSequence(h5_val, val_meta, 16384)

m_input = l.Input((1152, ))

m = l.BatchNormalization()(m_input)

m = l.Dense(4096)(m)
m = l.Activation('relu')(m)
m = l.BatchNormalization()(m)
m = l.Dropout(0.4)(m)

m = l.Dense(4096)(m)
m = l.Activation('relu')(m)
m = l.BatchNormalization()(m)
m = l.Dropout(0.4)(m)

m = l.Dense(4096)(m)
m = l.Activation('relu')(m)
m = l.BatchNormalization()(m)
m = l.Dropout(0.4)(m)

m = l.Dense(1000)(m)
m = l.Activation('sigmoid')(m)

mdl = k.models.Model(m_input, m)

print(mdl.summary())

mdl.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')

mdl.fit_generator(train_seq, epochs=10, validation_data=val_seq)

k.backend.set_value(mdl.optimizer.lr, 0.0001)

# Continue training from 10 epochs
# loss: 0.0022 - acc: 0.9994 - val_loss: 0.0021 - val_acc: 0.9994
mdl.fit_generator(train_seq, epochs=5, validation_data=val_seq)

mdl.save('basic_model_15epoch.keras')