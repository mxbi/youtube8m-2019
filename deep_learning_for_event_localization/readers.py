# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Provides readers configured for different datasets."""

import tensorflow as tf
import utils
import numpy as np


def gen_label_matrix(label, label_start, is_valid):
    output = np.zeros((60, 3862), dtype=np.bool)
    for lab, ls, val_l in zip(label, label_start//5, is_valid):
        output[ls, lab] = val_l
    return output


def gen_labels_matrix_flat(label, label_start, is_valid, seq_len):
    n_elements = seq_len // 5
    output = np.zeros((n_elements, 3862), dtype=np.bool)
    for lab, ls, val_l in zip(label, label_start // 5, is_valid):
        if ls < n_elements:  # This should not be the case!
            output[ls, lab] = val_l
    return output

def reshape_input(in_matrix, seq_len):
    outshape = seq_len // 5
    out_matrix = np.zeros((outshape, 5, 1152), np.float32)
    for i in range(outshape):
        out_matrix[i] = in_matrix[i*5:i*5+5]
    return out_matrix

def gen_label_mask(seq_len):
    mask = np.zeros((60), dtype=np.bool)
    mask[:seq_len] = 1
    return mask

def gen_has_label(is_valid, label_start):
    output = np.zeros((300), dtype=np.bool)
    for label_val, s_idx in zip(is_valid, label_start):
        if label_val:
            output[s_idx:s_idx+5] = 1
    return output

def gen_masked_has_label(is_valid, label_start):
    output = np.zeros((300), dtype=np.int8)
    for s_idx in label_start[is_valid == 0]:
        output[s_idx:s_idx + 5] = -1
    for s_idx in label_start[is_valid == 1]:
        output[s_idx:s_idx + 5] = 1
    return output

def gen_masked_select_one(is_valid, label_start, label_type):
    output = np.zeros((300), dtype=np.int8)
    selected_label = np.random.choice(np.unique(label_type))
    select_idx = label_type == selected_label
    is_valid = is_valid[select_idx]
    label_start = label_start[select_idx]

    for s_idx in label_start[is_valid == 0]:
        output[s_idx:s_idx + 5] = -1
    for s_idx in label_start[is_valid == 1]:
        output[s_idx:s_idx + 5] = 1
    return output, np.array(selected_label).reshape((1,))


def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be cast
      to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized


class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


class YT8MAggregatedFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """

  def __init__(  # pylint: disable=dangerous-default-value
      self,
      num_classes=3862,
      feature_sizes=[1024, 128],
      feature_names=["mean_rgb", "mean_audio"]):
    """Construct a YT8MAggregatedFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """

    assert len(feature_names) == len(feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(feature_names), len(feature_sizes)))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names

  def prepare_reader(self, filename_queue, batch_size=1024):
    """Creates a single reader thread for pre-aggregated YouTube 8M Examples.

    Args:
      filename_queue: A tensorflow queue of filename locations.
      batch_size: batch size used for feature output.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read_up_to(filename_queue, batch_size)

    tf.add_to_collection("serialized_examples", serialized_examples)
    return self.prepare_serialized_examples(serialized_examples)

  def prepare_serialized_examples(self, serialized_examples):
    """Parse a single video-level TF Example."""
    # set the mapping from the fields to data types in the proto
    num_features = len(self.feature_names)
    assert num_features > 0, "self.feature_names is empty!"
    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format(
        len(self.feature_names), len(self.feature_sizes))

    feature_map = {
        "id": tf.FixedLenFeature([], tf.string),
        "labels": tf.VarLenFeature(tf.int64)
    }
    for feature_index in range(num_features):
      feature_map[self.feature_names[feature_index]] = tf.FixedLenFeature(
          [self.feature_sizes[feature_index]], tf.float32)

    features = tf.parse_example(serialized_examples, features=feature_map)
    labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
    labels.set_shape([None, self.num_classes])
    concatenated_features = tf.concat(
        [features[feature_name] for feature_name in self.feature_names], 1)

    output_dict = {
        "video_ids": features["id"],
        "video_matrix": concatenated_features,
        "labels": labels,
        "num_frames": tf.ones([tf.shape(serialized_examples)[0]])
    }

    return output_dict


class YT8MFrameFeatureReader(BaseReader):
  """Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  """

  def __init__(  # pylint: disable=dangerous-default-value
      self,
      num_classes=3862,
      feature_sizes=[1024, 128],
      feature_names=["rgb", "audio"],
      max_frames=300,
      segment_labels=False,
      segment_size=5,
      prepare_distill=False):
    """Construct a YT8MFrameFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
      max_frames: the maximum number of frames to process.
      segment_labels: if we read segment labels instead.
      segment_size: the segment_size used for reading segments.
    """

    assert len(feature_names) == len(feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(feature_names), len(feature_sizes)))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names
    self.max_frames = max_frames
    self.segment_labels = segment_labels
    self.segment_size = segment_size

    self.prepare_distill = prepare_distill


  def get_video_matrix(self, features, feature_size, max_frames,
                       max_quantized_value, min_quantized_value):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
      num_frames: number of frames in the sequence
    """
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = utils.Dequantize(decoded_features, max_quantized_value,
                                      min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    if self.prepare_distill:
        def_feature_matrix = tf.reshape(tf.decode_raw(features, tf.uint8), [-1, feature_size])
        def_feature_matrix = resize_axis(def_feature_matrix, 0, max_frames)
        return feature_matrix, num_frames, def_feature_matrix

    return feature_matrix, num_frames

  def prepare_reader(self,
                     filename_queue,
                     max_quantized_value=2,
                     min_quantized_value=-2):
    """Creates a single reader thread for YouTube8M SequenceExamples.

    Args:
      filename_queue: A tensorflow queue of filename locations.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A tuple of video indexes, video features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    return self.prepare_serialized_examples(serialized_example,
                                            max_quantized_value,
                                            min_quantized_value)

  def prepare_serialized_examples(self,
                                  serialized_example,
                                  max_quantized_value=2,
                                  min_quantized_value=-2):
    """Parse single serialized SequenceExample from the TFRecords."""

    # Read/parse frame/segment-level labels.
    context_features = {
        "id": tf.FixedLenFeature([], tf.string),
    }
    if self.segment_labels:
      context_features.update({
          # There is no need to read end-time given we always assume the segment
          # has the same size.
          "segment_labels": tf.VarLenFeature(tf.int64),
          "segment_start_times": tf.VarLenFeature(tf.int64),
          "segment_scores": tf.VarLenFeature(tf.float32)
      })
    else:
      context_features.update({"labels": tf.VarLenFeature(tf.int64)})
    sequence_features = {
        feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string)
        for feature_name in self.feature_names
    }
    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features)
    if not self.segment_labels:
      labels = (tf.cast(
          tf.sparse_to_dense(contexts["labels"].values, (self.num_classes,), 1,
              validate_indices=False),
          tf.bool))

    # loads (potentially) different types of features and concatenates them
    num_features = len(self.feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(self.feature_names) == len(self.feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(self.feature_names), len(self.feature_sizes)))

    num_frames = -1  # the number of frames in the video
    feature_matrices = [None] * num_features  # an array of different features
    def_feature_matrices = [None] * num_features
    for feature_index in range(num_features):
      if self.prepare_distill:
        feature_matrix, num_frames_in_this_feature, def_fm = self.get_video_matrix(
          features[self.feature_names[feature_index]],
          self.feature_sizes[feature_index],
          self.max_frames,
          max_quantized_value,
          min_quantized_value)
      else:
        feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
            features[self.feature_names[feature_index]],
            self.feature_sizes[feature_index], self.max_frames,
            max_quantized_value, min_quantized_value)
      if num_frames == -1:
        num_frames = num_frames_in_this_feature

      feature_matrices[feature_index] = feature_matrix
      if self.prepare_distill:
        def_feature_matrices[feature_index] = def_fm

    # cap the number of frames at self.max_frames
    num_frames = tf.minimum(num_frames, self.max_frames)

    # concatenate different features
    video_matrix = tf.concat(feature_matrices, 1)

    if self.prepare_distill:
        batch_video_matrix = tf.expand_dims(video_matrix, 0)
        batch_labels = tf.expand_dims(labels, 0)
        batch_frames = tf.expand_dims(num_frames, 0)

        def_video_matrix = tf.concat(def_feature_matrices, 1)
        def_batch_video_matrix = tf.expand_dims(def_video_matrix, 0)
        batch_video_ids = tf.expand_dims(contexts["id"], 0)
        return batch_video_ids, batch_video_matrix, batch_labels, batch_frames, def_batch_video_matrix


    # Partition frame-level feature matrix to segment-level feature matrix.
    if self.segment_labels:
      start_times = contexts["segment_start_times"].values
      # Here we assume all the segments that started at the same start time has
      # the same segment_size.
      uniq_start_times, seg_idxs = tf.unique(
          start_times, out_idx=tf.dtypes.int64)
      # TODO(zhengxu): Ensure the segment_sizes are all same.
      segment_size = self.segment_size
      # Range gather matrix, e.g., [[0,1,2],[1,2,3]] for segment_size == 3.
      range_mtx = tf.expand_dims(
          uniq_start_times, axis=-1) + tf.expand_dims(
              tf.range(0, segment_size, dtype=tf.int64), axis=0)
      # Shape: [num_segment, segment_size, feature_dim].
      batch_video_matrix = tf.gather_nd(video_matrix,
                                        tf.expand_dims(range_mtx, axis=-1))
      num_segment = tf.shape(batch_video_matrix)[0]
      batch_video_ids = tf.reshape(
          tf.tile([contexts["id"]], [num_segment]), (num_segment,))
      batch_frames = tf.reshape(
          tf.tile([segment_size], [num_segment]), (num_segment,))

      # For segment labels, all labels are not exhausively rated. So we only
      # evaluate the rated labels.

      # Label indices for each segment, shape: [num_segment, 2].
      label_indices = tf.stack([seg_idxs, contexts["segment_labels"].values],
                               axis=-1)
      label_values = contexts["segment_scores"].values
      sparse_labels = tf.sparse.SparseTensor(label_indices, label_values,
                                             (num_segment, self.num_classes))
      batch_labels = tf.sparse.to_dense(sparse_labels, validate_indices=False)

      sparse_label_weights = tf.sparse.SparseTensor(
          label_indices, tf.ones_like(label_values, dtype=tf.float32),
          (num_segment, self.num_classes))
      batch_label_weights = tf.sparse.to_dense(
          sparse_label_weights, validate_indices=False)
    else:
      # Process video-level labels.
      label_indices = contexts["labels"].values
      sparse_labels = tf.sparse.SparseTensor(
          tf.expand_dims(label_indices, axis=-1),
          tf.ones_like(contexts["labels"].values, dtype=tf.bool),
          (self.num_classes,))
      labels = tf.sparse.to_dense(
          sparse_labels, default_value=False, validate_indices=False)
      # convert to batch format.
      batch_video_ids = tf.expand_dims(contexts["id"], 0)
      batch_video_matrix = tf.expand_dims(video_matrix, 0)
      batch_labels = tf.expand_dims(labels, 0)
      batch_frames = tf.expand_dims(num_frames, 0)
      batch_label_weights = None

    output_dict = {
        "video_ids": batch_video_ids,
        "video_matrix": batch_video_matrix,
        "labels": batch_labels,
        "num_frames": batch_frames,
    }
    if batch_label_weights is not None:
      output_dict["label_weights"] = batch_label_weights

    return output_dict


class YT8MValFrameFeatureReader(BaseReader):
  """Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  """

  def __init__(  # pylint: disable=dangerous-default-value
      self,
      num_classes=3862,
      feature_sizes=[1024, 128],
      feature_names=["rgb", "audio"],
      max_frames=300,
      segment_labels=True,
      segment_size=5,
      frame_it=False,
      label_presence=False,
      mask_unlabeled=False,
      one_prop=False):
    """Construct a YT8MFrameFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
      max_frames: the maximum number of frames to process.
      segment_labels: if we read segment labels instead.
      segment_size: the segment_size used for reading segments.
    """

    assert len(feature_names) == len(feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(feature_names), len(feature_sizes)))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names
    self.max_frames = max_frames
    self.segment_labels = segment_labels
    self.segment_size = segment_size
    self.frame_it = frame_it
    self.label_presence = label_presence
    self.mask_unlabeled = mask_unlabeled
    self.one_prop = one_prop

  def get_video_matrix(self, features, feature_size, max_frames,
                       max_quantized_value, min_quantized_value):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
      num_frames: number of frames in the sequence
    """
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = utils.Dequantize(decoded_features, max_quantized_value,
                                      min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)

    return feature_matrix, num_frames

  def prepare_reader(self,
                     filename_queue,
                     max_quantized_value=2,
                     min_quantized_value=-2):
    """Creates a single reader thread for YouTube8M SequenceExamples.

    Args:
      filename_queue: A tensorflow queue of filename locations.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A tuple of video indexes, video features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    return self.prepare_serialized_examples(serialized_example,
                                            max_quantized_value,
                                            min_quantized_value)

  def prepare_serialized_examples(self,
                                  serialized_example,
                                  max_quantized_value=2,
                                  min_quantized_value=-2):
    """Parse single serialized SequenceExample from the TFRecords."""

    # Read/parse frame/segment-level labels.
    context_features = {
        "id": tf.FixedLenFeature([], tf.string),
    }
    if self.segment_labels:
      context_features.update({
          # There is no need to read end-time given we always assume the segment
          # has the same size.
          "segment_labels": tf.VarLenFeature(tf.int64),
          "segment_start_times": tf.VarLenFeature(tf.int64),
          "segment_scores": tf.VarLenFeature(tf.float32)
      })
    else:
      context_features.update({"labels": tf.VarLenFeature(tf.int64)})
    sequence_features = {
        feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string)
        for feature_name in self.feature_names
    }
    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features)
    if not self.segment_labels:
      labels = (tf.cast(
          tf.sparse_to_dense(contexts["labels"].values, (self.num_classes,), 1,
              validate_indices=False),
          tf.bool))

    # loads (potentially) different types of features and concatenates them
    num_features = len(self.feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(self.feature_names) == len(self.feature_sizes), (
        "length of feature_names (={}) != length of feature_sizes (={})".format(
            len(self.feature_names), len(self.feature_sizes)))

    num_frames = -1  # the number of frames in the video
    feature_matrices = [None] * num_features  # an array of different features
    def_feature_matrices = [None] * num_features
    for feature_index in range(num_features):
      feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
            features[self.feature_names[feature_index]],
            self.feature_sizes[feature_index], self.max_frames,
            max_quantized_value, min_quantized_value)
      if num_frames == -1:
        num_frames = num_frames_in_this_feature

      feature_matrices[feature_index] = feature_matrix

    # cap the number of frames at self.max_frames
    num_frames = tf.minimum(num_frames, self.max_frames)

    # concatenate different features
    video_matrix = tf.concat(feature_matrices, 1)

    # New stuff here
    if self.label_presence:
        if self.one_prop:
            gen_has_label = gen_masked_select_one
        elif self.mask_unlabeled:
            gen_has_label = gen_masked_has_label

        else:
            global gen_has_label
            gen_has_label = gen_has_label

        if self.one_prop:
            labels, embedding = tf.py_func(gen_has_label,
                                  [contexts["segment_scores"].values, contexts["segment_start_times"].values,
                                   contexts["segment_labels"].values],
                                  [tf.int8, tf.int64])
            embedding.set_shape((1,))
            batch_embedding = tf.expand_dims(embedding, 0)

        else:
            labels = tf.py_func(gen_has_label,
                                [contexts["segment_scores"].values, contexts["segment_start_times"].values],
                                tf.int8 if self.mask_unlabeled else tf.bool)
        labels.set_shape((300,))
        batch_labels = tf.expand_dims(labels, 0)
        batch_video_ids = tf.expand_dims(contexts["id"], 0)
        batch_video_matrix = tf.expand_dims(video_matrix, 0)
        batch_frames = tf.expand_dims(num_frames, 0)

    elif self.frame_it:
        batch_video_matrix = tf.py_func(reshape_input, [video_matrix, num_frames], tf.float32)
        batch_video_matrix.set_shape((None, 5,1024+128))

        labels = tf.py_func(gen_labels_matrix_flat,
                        [
                            contexts["segment_labels"].values,
                            contexts["segment_start_times"].values,
                            contexts["segment_scores"].values,
                            num_frames
                        ],
                        tf.bool)
        labels.set_shape((None, 3862))
        batch_labels = labels
        batch_shape = tf.shape(labels)[0]
        batch_frames = tf.reshape(tf.tile([5], [batch_shape]), (batch_shape,))
        batch_video_ids = tf.expand_dims(contexts["id"], 0)
        batch_video_ids = tf.tile(batch_video_ids, [tf.shape(batch_frames)[0]])

    else:
      labels = tf.py_func(gen_label_matrix,
                        [
                            contexts["segment_labels"].values,
                            contexts["segment_start_times"].values,
                            contexts["segment_scores"].values],
                        tf.bool)
      labels.set_shape((60, 3862))
      batch_labels = tf.expand_dims(labels, 0)

      mask = tf.py_func(gen_label_mask, [num_frames], tf.bool)
      mask.set_shape((60,))
      batch_video_matrix = tf.expand_dims(video_matrix, 0)
      batch_labels = tf.expand_dims(labels, 0)
      batch_frames = tf.expand_dims(num_frames, 0)
      batch_mask = tf.expand_dims(mask, 0)
      batch_video_ids = tf.expand_dims(contexts["id"], 0)  # TODO: fix it


    output_dict = {
        # "video_mask": batch_mask,
        "video_ids": batch_video_ids,
        "video_matrix": batch_video_matrix,
        "labels": batch_labels,
        "num_frames": batch_frames,
    }
    if self.one_prop:
        output_dict["sel_label"] = batch_embedding
    return output_dict
