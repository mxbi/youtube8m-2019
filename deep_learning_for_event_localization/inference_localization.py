# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary for generating predictions over a set of videos."""

from __future__ import print_function

import glob
import heapq
import json
import os
import tarfile
import tempfile
import time
import numpy as np

import readers
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.lib.io import file_io
import utils

# Torch stuff
import torch.nn as nn
import torch.optim as optim
import torch
from torch.nn.utils.rnn import pack_padded_sequence

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Input
  flags.DEFINE_string(
      "train_dir", "", "The directory to load the model files from. We assume "
      "that you have already run eval.py onto this, such that "
      "inference_model.* files already exist.")
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string(
      "input_model_tgz", "",
      "If given, must be path to a .tgz file that was written "
      "by this binary using flag --output_model_tgz. In this "
      "case, the .tgz file will be untarred to "
      "--untar_model_dir and the model will be used for "
      "inference.")
  flags.DEFINE_string(
      "untar_model_dir", "/tmp/yt8m-model",
      "If --input_model_tgz is given, then this directory will "
      "be created and the contents of the .tgz file will be "
      "untarred here.")

  flags.DEFINE_bool(
      "segment_labels", False,
      "If set, then --input_data_pattern must be frame-level features (but with"
      " segment_labels). Otherwise, --input_data_pattern must be aggregated "
      "video-level features. The model must also be set appropriately (i.e. to "
      "read 3D batches VS 4D batches.")

  flags.DEFINE_integer("segment_max_pred", 100000,
                       "Limit total number of segment outputs per entity.")
  flags.DEFINE_string("segment_label_ids_file", "segment_label_ids.csv",
                      "The file that contains the segment label ids.")

  # Output
  flags.DEFINE_string("output_file", "", "The file to save the predictions to.")
  flags.DEFINE_string(
      "output_model_tgz", "",
      "If given, should be a filename with a .tgz extension, "
      "the model graph and checkpoint will be bundled in this "
      "gzip tar. This file can be uploaded to Kaggle for the "
      "top 10 participants.")
  flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")

  # Other flags.
  flags.DEFINE_integer("batch_size", 8192,
                       "How many examples to process per batch.")
  flags.DEFINE_integer("num_readers", 1,
                       "How many threads to use for reading input files.")

  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                                                   "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  flags.DEFINE_string(
      "model", "LogisticModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")
  flags.DEFINE_string(
      "checkpoint", "XXX",
      "Checkpoint to load the model from!")
  flags.DEFINE_string("prediction_file", "",
                      "Creates files for prediction. Does not create submission file.")
  flags.DEFINE_integer("embedd", 0,
                       "embedding_value")
  flags.DEFINE_integer("device", 0,
                       "CUDA device")

class BiLSTM_model(nn.Module):
    """
    Torch model
    """
    def __init__(self, in_shape=1152, hidden_size=256, max_len=300, embedd_size=32):
        super(BiLSTM_model, self).__init__()
        self.lstm = nn.LSTM(in_shape + embedd_size, hidden_size,
                            batch_first=True, num_layers=2,
                            bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.max_len = max_len
        self.linear = nn.Linear(hidden_size * 2, 1)
        self.embed = torch.nn.Embedding(1000, embedd_size)

    def forward(self, x, input_lengths, mask, embeddings):
        ebedd = self.embed(embeddings)
        ebedd = ebedd.view(ebedd.shape[0], ebedd.shape[1], 1).repeat(1, 1, 300)
        ebedd = torch.transpose(ebedd, 1, 2)
        x = torch.cat([x, ebedd], dim=2)

        x = pack_padded_sequence(x, input_lengths, batch_first=True)


        out = self.lstm(x)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out[0], batch_first=True, total_length=self.max_len)
        return self.sigmoid(self.linear(unpacked).squeeze()) * mask


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def format_lines(video_ids, predictions, top_k, whitelisted_cls_mask=None):
  """Create an information line the submission file."""
  batch_size = len(video_ids)
  for video_index in range(batch_size):
    video_prediction = predictions[video_index]
    if whitelisted_cls_mask is not None:
      # Whitelist classes.
      video_prediction *= whitelisted_cls_mask
    top_indices = np.argpartition(video_prediction, -top_k)[-top_k:]
    line = [(class_index, predictions[video_index][class_index])
            for class_index in top_indices]
    line = sorted(line, key=lambda p: -p[1])
    yield video_ids[video_index].decode("utf-8") + "," + " ".join(
        "%i %g" % (label, score) for (label, score) in line) + "\n"


def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.

  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    examples_and_labels = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    input_data_dict = (
        tf.train.batch_join(
            examples_and_labels,
            batch_size=batch_size,
            allow_smaller_final_batch=True,
            enqueue_many=True))
    video_id_batch = input_data_dict["video_ids"]
    video_batch = input_data_dict["video_matrix"]
    num_frames_batch = input_data_dict["num_frames"]
    return video_id_batch, video_batch, num_frames_batch


def build_data_providers(reader,
                train_data_pattern,
                batch_size=1000,
                num_readers=1,
                num_epochs=None):
  input_data_dict = (
      get_input_data_tensors(
          reader,
          train_data_pattern,
          batch_size=batch_size,
          num_readers=num_readers))
  return input_data_dict

def inference(reader, train_dir, data_pattern, out_file_location, batch_size,
              top_k):
  """Inference function."""
  all_preds = []
  all_ids = []

  data_dict = build_data_providers(reader=readers.YT8MValFrameFeatureReader(label_presence=True),
                                   train_data_pattern=data_pattern,
                                   batch_size=batch_size,
                                   num_readers=1,
                                   num_epochs=1)

  with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True)) as sess:


    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    sess.run(
        set_up_init_ops(tf.get_collection_ref(tf.GraphKeys.LOCAL_VARIABLES)))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Torch model setup
    torch.cuda.set_device(FLAGS.device)
    # models = [BiLSTM_model(hidden_size=256) for i in range(5)]
    models = [BiLSTM_model(hidden_size=256)]
    #for i, model in enumerate(models):
    # model.load_state_dict(torch.load("/mnt/4tbyoutube/model/torched_masking_unlabaled/torch_model_3epochs_256hidden_run{}.pth".format(i+1)))
    models[0].load_state_dict(torch.load("/mnt/4tbyoutube/shared/torched_embedd/torch_model_3epochs_256hidden.pth"))
    models[0].cuda(FLAGS.device)
    models[0].eval()

    num_examples_processed = 0


    try:
      while not coord.should_stop():

        data = sess.run(data_dict)

        x = data[1]
        x_len = data[2]
        x_ids = data[0]

        # Preprocess the sequences
        soder = np.argsort(x_len)[::-1]
        x = x[soder]
        x_len = x_len[soder]
        x_ids = x_ids[soder]
        mask = np.zeros((x.shape[0], 300), np.bool)

        embedds = torch.zeros(x_len.shape, dtype=torch.int64).cuda(FLAGS.device) + FLAGS.embedd

        for i, slen in enumerate(x_len):
            mask[i, :slen] = 1

        with torch.no_grad():
            x_input = torch.tensor(x, device='cuda:{}'.format(FLAGS.device))
            mask = torch.tensor(mask, device='cuda:{}'.format(FLAGS.device), dtype=torch.float)
            predictions_val = np.mean([model.forward(x_input, x_len, mask, embedds).cpu().numpy() for model in models], axis=0)
            # predictions_val = model.forward(x_input, x_len, mask).cpu().numpy()

        all_preds.append(predictions_val)
        all_ids.append(x_ids)

        num_examples_processed += x.shape[0]
        logging.info("Processed: " + str(num_examples_processed))

    except tf.errors.OutOfRangeError:
      logging.info("Done with inference. The output file was written to " + FLAGS.prediction_file)
    finally:
      coord.request_stop()

      all_preds = np.concatenate(all_preds, axis=0)
      all_ids = np.concatenate(all_ids, axis=0)
      np.save(FLAGS.prediction_file + '_{}_preds.npy'.format(FLAGS.embedd), all_preds)
      np.save(FLAGS.prediction_file + '_{}_ids.npy'.format(FLAGS.embedd), all_ids)

    coord.join(threads)
    sess.close()


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.input_data_pattern:
    raise ValueError("'input_data_pattern' was not specified. "
                     "Unable to continue with inference.")

  inference(None, FLAGS.train_dir, FLAGS.input_data_pattern,
            FLAGS.output_file, FLAGS.batch_size, FLAGS.top_k)


if __name__ == "__main__":
  app.run()
