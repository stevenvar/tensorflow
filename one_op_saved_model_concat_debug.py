import argparse
import os
import shutil

import numpy as np
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()

_BATCH_SIZE = 3
_TIME_STEPS = 2
_CONCAT_WIDTH = 8
_FINAL_CONCAT_WIDTH = 32


def build_concat_debug_model():
  """Builds a small subgraph for concat-debug reproduction."""
  concat_a = tf.placeholder(
      dtype=tf.float32,
      shape=[_BATCH_SIZE, _TIME_STEPS, _CONCAT_WIDTH],
      name="concat_a")
  concat_b = tf.placeholder(
      dtype=tf.float32,
      shape=[_BATCH_SIZE, _TIME_STEPS, _CONCAT_WIDTH],
      name="concat_b")
  concat_c = tf.placeholder(
      dtype=tf.float32,
      shape=[_BATCH_SIZE, _TIME_STEPS, _CONCAT_WIDTH],
      name="concat_c")
  expanddims_input = tf.placeholder(
      dtype=tf.float32,
      shape=[_BATCH_SIZE, _TIME_STEPS],
      name="expanddims_input")
  final_concat_lhs = tf.placeholder(
      dtype=tf.float32,
      shape=[_BATCH_SIZE, _FINAL_CONCAT_WIDTH],
      name="final_concat_lhs")

  with tf.name_scope("branch"):
    concat_3d = tf.concat(
        [concat_a, concat_b, concat_c], axis=-1, name="concat")

    expanded = tf.expand_dims(
        expanddims_input, axis=2, name="ExpandDims")

    slice_end = tf.stack(
        [
            tf.shape(expanded)[0],
            tf.shape(expanded)[1],
            tf.constant(1, dtype=tf.int32),
        ],
        axis=0,
        name="Pack")
    strided = tf.strided_slice(
        expanded,
        begin=tf.constant([0, 0, 0], dtype=tf.int32, name="begin"),
        end=slice_end,
        strides=tf.constant([1, 1, 1], dtype=tf.int32, name="strides"),
        name="StridedSlice")

    gated = tf.multiply(concat_3d, strided, name="mul")
    reduced = tf.reduce_sum(
        gated,
        axis=1,
        keepdims=False,
        name="Sum")

    scale = tf.constant(
        np.linspace(0.25, 1.75, 24, dtype=np.float32), name="scale_const")
    branch_output = tf.multiply(reduced, scale, name="scaled_output")

  final_concat = tf.concat(
      [final_concat_lhs, branch_output], axis=1, name="final_concat")

  feeds = {
      concat_a: np.arange(
          _BATCH_SIZE * _TIME_STEPS * _CONCAT_WIDTH,
          dtype=np.float32).reshape(_BATCH_SIZE, _TIME_STEPS, _CONCAT_WIDTH),
      concat_b: (100 + np.arange(
          _BATCH_SIZE * _TIME_STEPS * _CONCAT_WIDTH,
          dtype=np.float32)).reshape(_BATCH_SIZE, _TIME_STEPS, _CONCAT_WIDTH),
      concat_c: (200 + np.arange(
          _BATCH_SIZE * _TIME_STEPS * _CONCAT_WIDTH,
          dtype=np.float32)).reshape(_BATCH_SIZE, _TIME_STEPS, _CONCAT_WIDTH),
      expanddims_input: np.array(
          [[1.0, 0.5], [0.25, 1.5], [2.0, 1.0]], dtype=np.float32),
      final_concat_lhs: np.arange(
          _BATCH_SIZE * _FINAL_CONCAT_WIDTH,
          dtype=np.float32).reshape(_BATCH_SIZE, _FINAL_CONCAT_WIDTH),
  }
  inputs = {
      "concat_a": concat_a,
      "concat_b": concat_b,
      "concat_c": concat_c,
      "expanddims_input": expanddims_input,
      "final_concat_lhs": final_concat_lhs,
  }
  outputs = {
      "branch_output": branch_output,
      "final_concat": final_concat,
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
          "Generate a TensorFlow v1 SavedModel for the concat_debug "
          "subgraph (Concat -> Mul -> Sum -> Mul)."))
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
        print("branch_output shape:", fetched["branch_output"].shape)
        print("final_concat shape:", fetched["final_concat"].shape)

  print("Done")


if __name__ == "__main__":
  main()
