import argparse
import os
import shutil

import numpy as np
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()

_BATCH_SIZE = 3
_LEFT_SEQUENCE_WIDTH = 240
_LEFT_EMBED_WIDTH = 4
_RIGHT_ACT_WIDTH = 4
_RIGHT_ATTR_WIDTH = 20
_RIGHT_TOTAL_WIDTH = _RIGHT_ACT_WIDTH + _RIGHT_ATTR_WIDTH


def build_concat_debug_model():
  """Builds a reduced repro of the concat/truediv shape pattern."""
  sparse_features_flat_a = tf.placeholder(
      dtype=tf.float32, shape=[None], name="sparse_features_flat_a")
  sparse_features_flat_b = tf.placeholder(
      dtype=tf.float32, shape=[None], name="sparse_features_flat_b")
  sequence_mask = tf.placeholder(
      dtype=tf.float32,
      shape=[None, _LEFT_SEQUENCE_WIDTH, _LEFT_EMBED_WIDTH],
      name="sequence_mask")

  act_padding_flat = tf.placeholder(
      dtype=tf.float32, shape=[None], name="act_padding_flat")
  attr_padding_flat = tf.placeholder(
      dtype=tf.float32, shape=[None], name="attr_padding_flat")
  gating_input = tf.placeholder(
      dtype=tf.float32, shape=[None, None], name="gating_input")

  with tf.name_scope("BuildCommonEmbInput"):
    sparse_features_concat = tf.concat(
        [sparse_features_flat_a, sparse_features_flat_b],
        axis=0,
        name="sparse_features_embedding_concat")
    left_reshape = tf.reshape(
        sparse_features_concat,
        [-1, _LEFT_SEQUENCE_WIDTH, _LEFT_EMBED_WIDTH],
        name="Reshape")
    left_mul = tf.multiply(left_reshape, sequence_mask, name="mul")
    left_branch = tf.reshape(left_mul, [-1, 960], name="Reshape_2")

  with tf.name_scope("sequence_branch"):
    act_padding_session = tf.reshape(
        act_padding_flat, [-1, 10, _RIGHT_ACT_WIDTH], name="Reshape_5")
    attr_padding_session = tf.reshape(
        attr_padding_flat, [-1, 10, _RIGHT_ATTR_WIDTH], name="Reshape_6")
    concat_4 = tf.concat(
        [act_padding_session, attr_padding_session], axis=2, name="concat_4")
    expanded = tf.expand_dims(gating_input, axis=2, name="ExpandDims")
    slice_end = tf.stack(
        [
            tf.shape(expanded)[0],
            tf.shape(expanded)[1],
            tf.constant(1, dtype=tf.int32),
        ],
        axis=0,
        name="strided_slice_4_end")
    gating_slice = tf.strided_slice(
        expanded,
        begin=tf.constant([0, 0, 0], dtype=tf.int32, name="strided_slice_4_begin"),
        end=slice_end,
        strides=tf.constant([1, 1, 1], dtype=tf.int32, name="strided_slice_4_stride"),
        name="strided_slice_4")
    mul_1 = tf.multiply(concat_4, gating_slice, name="mul_1")
    sum_1 = tf.reduce_sum(mul_1, axis=1, keepdims=False, name="Sum_1")
    truediv_2 = tf.multiply(
        sum_1, tf.constant(1.0 / 10.0, dtype=tf.float32), name="truediv_2")

  final_concat = tf.concat(
      [left_branch, truediv_2], axis=1, name="input_emb_concat")

  feeds = {
      sparse_features_flat_a: np.arange(
          (_BATCH_SIZE * _LEFT_SEQUENCE_WIDTH * _LEFT_EMBED_WIDTH) // 2,
          dtype=np.float32),
      sparse_features_flat_b: (10000 + np.arange(
          (_BATCH_SIZE * _LEFT_SEQUENCE_WIDTH * _LEFT_EMBED_WIDTH) // 2,
          dtype=np.float32)),
      sequence_mask: np.linspace(
          0.25,
          1.25,
          _BATCH_SIZE * _LEFT_SEQUENCE_WIDTH * _LEFT_EMBED_WIDTH,
          dtype=np.float32).reshape(
              _BATCH_SIZE, _LEFT_SEQUENCE_WIDTH, _LEFT_EMBED_WIDTH),
      act_padding_flat: np.arange(
          _BATCH_SIZE * 10 * _RIGHT_ACT_WIDTH, dtype=np.float32),
      attr_padding_flat: (1000 + np.arange(
          _BATCH_SIZE * 10 * _RIGHT_ATTR_WIDTH, dtype=np.float32)),
      gating_input: np.array(
          [
              np.linspace(1.0, 2.0, 10, dtype=np.float32),
              np.linspace(0.25, 1.25, 10, dtype=np.float32),
              np.linspace(2.0, 3.0, 10, dtype=np.float32),
          ],
          dtype=np.float32),
  }
  inputs = {
      "sparse_features_flat_a": sparse_features_flat_a,
      "sparse_features_flat_b": sparse_features_flat_b,
      "sequence_mask": sequence_mask,
      "act_padding_flat": act_padding_flat,
      "attr_padding_flat": attr_padding_flat,
      "gating_input": gating_input,
  }
  outputs = {
      "left_branch": left_branch,
      "truediv_2": truediv_2,
      "input_emb_concat": final_concat,
  }
  return inputs, outputs, feeds


def build_signature(inputs, outputs):
  tensor_inputs = {
      key: tf.saved_model.utils.build_tensor_info(tensor)
      for key, tensor in inputs.items()
  }
  tensor_outputs = {
      key: tf.saved_model.utils.build_tensor_info(tensor)
      for key, tensor in outputs.items()
  }
  return tf.saved_model.signature_def_utils.build_signature_def(
      inputs=tensor_inputs,
      outputs=tensor_outputs,
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
  )


def save_model(sess, export_dir, signature):
  model_dir = os.path.dirname(export_dir)
  if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)

  builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
  builder.add_meta_graph_and_variables(
      sess,
      [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              signature
      },
  )
  builder.save()


def parse_args():
  parser = argparse.ArgumentParser(
      description=(
          "Generate a TensorFlow v1 SavedModel for a reduced concat debug "
          "subgraph based on BuildCommonEmbInput/Reshape_2 and truediv_2."))
  parser.add_argument(
      "--save_model",
      action="store_true",
      help="Save the model instead of executing it.")
  parser.add_argument(
      "--export_dir",
      default=os.path.join("unit_tests", "model_CONCAT_DEBUG", "1"),
      help="SavedModel export directory.")
  return parser.parse_args()


def main():
  args = parse_args()

  with tf.Graph().as_default():
    inputs, outputs, feeds = build_concat_debug_model()
    signature = build_signature(inputs, outputs)

    with tf.Session() as sess:
      if args.save_model:
        save_model(sess, args.export_dir, signature)
        print("Saved model to {}".format(args.export_dir))
      else:
        fetched = sess.run(outputs, feed_dict=feeds)
        print("left_branch shape:", fetched["left_branch"].shape)
        print("truediv_2 shape:", fetched["truediv_2"].shape)
        print("input_emb_concat shape:", fetched["input_emb_concat"].shape)

  print("Done")


if __name__ == "__main__":
  main()
