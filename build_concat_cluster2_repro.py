import argparse
import os
import shutil

import tensorflow.compat.v1 as tf


tf.disable_eager_execution()


def _tensor_info_map(tensors):
  return {
      name: tf.saved_model.utils.build_tensor_info(tensor)
      for name, tensor in tensors.items()
  }


def _build_signature(inputs, outputs):
  return tf.saved_model.signature_def_utils.build_signature_def(
      inputs=_tensor_info_map(inputs),
      outputs=_tensor_info_map(outputs),
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
  )


def _save_model(sess, export_dir, signature):
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


def build_graph():
  with tf.name_scope("model"):
    # A is carried by the first dimension of this input.
    common_flat = tf.placeholder(
        tf.float32, shape=[None, 4], name="common_flat")

    # Mimic BuildCommonEmbInput/Reshape -> mul -> Reshape_2.
    common_reshape = tf.reshape(
        common_flat, [-1, 240, 4], name="BuildCommonEmbInput/Reshape")
    common_scale = tf.reshape(tf.linspace(1.0, 4.0, 4), [1, 1, 4], name="Reshape")
    common_mul = tf.multiply(
        common_reshape, common_scale, name="BuildCommonEmbInput/mul")
    common_reshape_2 = tf.reshape(
        common_mul, [-1, 960], name="BuildCommonEmbInput/Reshape_2")

    # Core shape-to-value chain we want to stress.
    common_shape = tf.shape(common_reshape_2, out_type=tf.int32, name="Shape")
    first_dim_vec = tf.strided_slice(
        common_shape, [0], [1], [1], name="strided_slice_1")
    first_dim = tf.squeeze(first_dim_vec, axis=0, name="first_dim")
    tile_multiples = tf.stack(
        [first_dim, 1], axis=0, name="Tile/multiples")

    # A small arithmetic chain on the dynamic dimension value itself.
    first_dim_plus_one = tf.add(first_dim, 1, name="first_dim_plus_one")
    first_dim_times_24 = tf.multiply(
        first_dim, 24, name="first_dim_times_24")

    # Use the shape-derived value in a couple of consumers.
    base_row = tf.reshape(
        tf.linspace(1.0, 24.0, 24), [1, 24], name="base_row")
    tiled_row = tf.tile(base_row, tile_multiples, name="Tile_3")
    post_tile_bias = tf.add(
        tiled_row, tf.constant(0.5, dtype=tf.float32), name="post_tile_bias")

    reshape_shape = tf.stack(
        [first_dim_times_24, 40], axis=0, name="reshape_shape")
    reshaped_from_flat = tf.reshape(
        common_reshape_2, reshape_shape, name="debug_reshape")

    filled = tf.fill(
        reshape_shape, tf.constant(7.0, dtype=tf.float32), name="debug_fill")

  inputs = {
      "common_flat": common_flat,
  }
  outputs = {
      "common_reshape_2": common_reshape_2,
      "common_shape": common_shape,
      "first_dim_vec": first_dim_vec,
      "first_dim": first_dim,
      "first_dim_plus_one": first_dim_plus_one,
      "first_dim_times_24": first_dim_times_24,
      "tile_multiples": tile_multiples,
      "tiled_row": tiled_row,
      "post_tile_bias": post_tile_bias,
      "reshaped_from_flat": reshaped_from_flat,
      "filled": filled,
  }
  return inputs, outputs


def make_feeds(a_value):
  if a_value % 240 != 0:
    raise ValueError("A must be divisible by 240, got {}".format(a_value))

  common_flat = [
      [float(row * 4 + col) for col in range(4)] for row in range(a_value)
  ]
  return {
      "common_flat": common_flat,
  }


def _summarize_fetches(fetched):
  keys = [
      "common_reshape_2",
      "common_shape",
      "first_dim_vec",
      "first_dim",
      "first_dim_plus_one",
      "first_dim_times_24",
      "tile_multiples",
      "tiled_row",
      "post_tile_bias",
      "reshaped_from_flat",
      "filled",
  ]
  parts = []
  for name in keys:
    value = fetched[name]
    shape = tuple(value.shape) if hasattr(value, "shape") else ()
    parts.append("{}={}".format(name, shape))
  return ", ".join(parts)


def _run_once(sess, inputs, outputs, a_value):
  feeds = make_feeds(a_value)
  feed_dict = {inputs[name]: value for name, value in feeds.items()}
  fetched = sess.run(outputs, feed_dict=feed_dict)
  print("Executed graph with A={}: {}".format(
      a_value, _summarize_fetches(fetched)))
  print("  common_shape={}".format(fetched["common_shape"].tolist()))
  print("  first_dim_vec={}".format(fetched["first_dim_vec"].tolist()))
  print("  first_dim={}".format(int(fetched["first_dim"])))
  print("  first_dim_plus_one={}".format(int(fetched["first_dim_plus_one"])))
  print("  first_dim_times_24={}".format(int(fetched["first_dim_times_24"])))
  print("  tile_multiples={}".format(fetched["tile_multiples"].tolist()))
  print("  tiled_row_rows={}".format(fetched["tiled_row"].shape[0]))
  print("  post_tile_bias_rows={}".format(fetched["post_tile_bias"].shape[0]))
  print("  reshaped_from_flat_rows={}".format(fetched["reshaped_from_flat"].shape[0]))
  print("  filled_rows={}".format(fetched["filled"].shape[0]))
  return fetched


def parse_args():
  parser = argparse.ArgumentParser(
      description=(
          "Build a tiny SavedModel focused on the cluster_2 shape-to-value "
          "chain: reshape -> shape -> stridedslice -> arithmetic -> "
          "pack -> tile/reshape/fill."))
  parser.add_argument(
      "--output_dir",
      default="model_BESPOKE3/1",
      help="SavedModel export directory.")
  parser.add_argument(
      "--a_value",
      type=int,
      default=720,
      help="Concrete feed value for the symbolic dimension A. Must be divisible by 240.")
  parser.add_argument(
      "--sequence_a_values",
      default="",
      help=(
          "Optional comma-separated list of A values to run sequentially in "
          "the same session, for example '240,720'."))
  parser.add_argument(
      "--run_only",
      action="store_true",
      help="Run the graph once instead of exporting a SavedModel.")
  return parser.parse_args()


def main():
  args = parse_args()
  inputs, outputs = build_graph()
  signature = _build_signature(inputs, outputs)
  sequence_values = []
  if args.sequence_a_values:
    sequence_values = [
        int(value.strip()) for value in args.sequence_a_values.split(",")
        if value.strip()
    ]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if sequence_values:
      for index, a_value in enumerate(sequence_values):
        if index:
          print("---")
        _run_once(sess, inputs, outputs, a_value)
    else:
      _run_once(sess, inputs, outputs, args.a_value)

    if not args.run_only:
      _save_model(sess, args.output_dir, signature)
      print("Saved model to {}".format(args.output_dir))


if __name__ == "__main__":
  main()
