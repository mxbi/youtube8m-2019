"""This will get torched"""

import json
import os
import frame_level_models
import video_level_models
import readers
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import numpy as np

# Torch stuff
import torch.nn as nn
import torch.optim as optim
import torch
from torch.nn.utils.rnn import pack_padded_sequence

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to save the model files in.")
  flags.DEFINE_string(
      "train_data_pattern", "",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", False,
      "If set, then --train_data_pattern must be frame-level features. "
      "Otherwise, --train_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_bool(
      "segment_labels", False,
      "If set, then --train_data_pattern must be frame-level features (but with"
      " segment_labels). Otherwise, --train_data_pattern must be aggregated "
      "video-level features. The model must also be set appropriately (i.e. to "
      "read 3D batches VS 4D batches.")
  flags.DEFINE_string(
      "model", "LogisticModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")
  flags.DEFINE_bool(
      "start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")

  flags.DEFINE_bool(
      "mask", False,
      "If set, this will mask the output before loss computation. Useful for finetuning.")
  flags.DEFINE_bool(
      "full_seq", False,
      "If set, this will mask the output before loss computation. Useful for finetuning.")
  flags.DEFINE_bool(
      "mask_unlabeled", False,
      "This will compute loss only on annotated regions.")


  flags.DEFINE_string(
      "seed_weights", "",
      "Location of the checkpoint you want to import weights from.")

  # Training flags.
  flags.DEFINE_integer(
      "num_gpu", 1, "The maximum number of GPU devices to use for training. "
      "Flag only applies if GPUs are installed")
  flags.DEFINE_integer("batch_size", 1024,
                       "How many examples to process per batch for training.")
  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Which loss function to use for training the model.")
  flags.DEFINE_float(
      "regularization_penalty", 1.0,
      "How much weight to give to the regularization loss (the label loss has "
      "a weight of 1).")
  flags.DEFINE_float("base_learning_rate", 0.01,
                     "Which learning rate to start with.")
  flags.DEFINE_float(
      "learning_rate_decay", 0.95,
      "Learning rate decay factor to be applied every "
      "learning_rate_decay_examples.")
  flags.DEFINE_float(
      "learning_rate_decay_examples", 4000000,
      "Multiply current learning rate by learning_rate_decay "
      "every learning_rate_decay_examples.")
  flags.DEFINE_integer(
      "num_epochs", 5, "How many passes to make over the dataset before "
      "halting training.")
  flags.DEFINE_integer(
      "max_steps", None,
      "The maximum number of iterations of the training loop.")

  # Other flags.
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")
  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")
  flags.DEFINE_bool(
      "log_device_placement", False,
      "Whether to write the device on which every op will run into the "
      "logs on startup.")


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


def validate_class_name(flag_value, category, modules, expected_superclass):
  """Checks that the given string matches a class of the expected type.
  """
  candidates = [getattr(module, flag_value, None) for module in modules]
  for candidate in candidates:
    if not candidate:
      continue
    if not issubclass(candidate, expected_superclass):
      raise flags.FlagsError(
          "%s '%s' doesn't inherit from %s." %
          (category, flag_value, expected_superclass.__name__))
    return True
  raise flags.FlagsError("Unable to find %s '%s'." % (category, flag_value))


def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=8):
  """
  Creates the section of the graph which reads the training data.
  """
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    logging.info("#### Number of training files: %s.", str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=batch_size * 10,
        min_after_dequeue=batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


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
          num_readers=num_readers,
          num_epochs=num_epochs))
  return input_data_dict

class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self,
               cluster,
               task,
               train_dir,
               model,
               reader,
               model_exporter,
               log_device_placement=True,
               max_steps=None):
    """"Creates a Trainer."""

    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.train_dir = train_dir
    self.config = tf.ConfigProto(
        allow_soft_placement=True,
        device_count={'GPU': 0},
        log_device_placement=log_device_placement)
    self.config.gpu_options.allow_growth = True
    self.model = model
    self.reader = reader
    self.model_exporter = model_exporter
    self.max_steps = max_steps
    self.max_steps_reached = False
    self.last_model_export_step = 0


  def run(self):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    torch.cuda.set_device(0)
    model = BiLSTM_model(hidden_size=256)
    model.cuda(0)
    # model = nn.DataParallel(model)


    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    bcloss = nn.BCELoss(reduction='mean')

    if not os.path.exists(self.train_dir):
      os.makedirs(self.train_dir)
    with tf.device("/cpu:0"):
        data_dict = build_data_providers(self.reader,
                                         FLAGS.train_data_pattern,
                                         batch_size=FLAGS.batch_size,
                                         num_readers=FLAGS.num_readers,
                                         num_epochs=FLAGS.num_epochs)


        sv = tf.train.Supervisor(
            tf.get_default_graph(),
            logdir=self.train_dir,
            init_op=tf.global_variables_initializer())

        import pandas as pd
        df = pd.read_csv("./youtube-8m/segment_label_ids.csv")
        idx_val = {x: i for i, x in enumerate(df.Index)}
        step = 0
        with sv.managed_session("", config=self.config) as sess:
        # with tf.Session(config=self.config) as sess:
          try:
            logging.info("%s: Entering training loop.", task_as_string(self.task))
            while True: # (not sv.should_stop()) and (not self.max_steps_reached):
                step += 1
                data = sess.run(data_dict)
                x = data['video_matrix']
                y = data['labels']
                x_len = data['num_frames']

                label_type = np.array([idx_val[xx[0]] for xx in data['sel_label']])

                # Preprocess the sequences
                soder = np.argsort(x_len)[::-1]
                x = x[soder]
                y = y[soder]
                label_type = label_type[soder]

                x_len = x_len[soder]
                if FLAGS.mask_unlabeled:
                    mask = y != 0
                    y = y == 1
                else:
                    mask = np.zeros((x.shape[0], 300), np.bool)
                    for i, slen in enumerate(x_len):
                        mask[i, :slen] = 1

                # model processing
                optimizer.zero_grad()
                y_input = torch.tensor(y, device='cuda:0', dtype=torch.float)
                x_input = torch.tensor(x, device='cuda:0')
                mask = torch.tensor(mask, device='cuda:0', dtype=torch.float)
                label_type = torch.tensor(label_type, device='cuda:0', dtype=torch.int64)

                preds = model.forward(x_input, x_len, mask, label_type)

                my_loss = bcloss(preds, y_input * mask)
                my_loss.backward()
                optimizer.step()
                closs = float(my_loss.data.cpu().numpy())
                print("step: {}, loss: {:.5f}".format(step, closs))

                if step == 92:
                    print("Reducing LR")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.0001
                if step == 92*2:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.00002
                if step == 92*3:
                    torch.save(model.state_dict(), os.path.join(self.train_dir, "torch_model_3epochs_256hidden.pth"))
                    exit()

                    # print("Reducing LR")
                    # for param_group in optimizer.param_groups:
                    #     param_group['lr'] = 0.00002

                if step == 92*12:
                    print("We are done.")
                    torch.save(model.state_dict(), os.path.join(self.train_dir, "torch_model_longer_wider.pth"))
                    exit()

          except tf.errors.OutOfRangeError:
            logging.info("%s: Done training -- epoch limit reached.",
                         task_as_string(self.task))


    logging.info("%s: Exited training loop.", task_as_string(self.task))
    sv.Stop()


def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)


def main(unused_argv):
  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  # Logging the version.
  logging.set_verbosity(tf.logging.INFO)
  logging.info("%s: Tensorflow version: %s.", task_as_string(task),
               tf.__version__)

  # Dispatch to a master, a worker, or a parameter server.
  if not cluster or task.type == "master" or task.type == "worker":
    model = find_class_by_name(FLAGS.model,
                               [frame_level_models, video_level_models])()

    reader = readers.YT8MValFrameFeatureReader(label_presence=True, mask_unlabeled=FLAGS.mask_unlabeled,
                                               one_prop=True)

    Trainer(cluster, task, FLAGS.train_dir, model, reader, None,
            FLAGS.log_device_placement, FLAGS.max_steps).run()

if __name__ == "__main__":
  app.run()
