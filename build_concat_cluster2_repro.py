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
    # This placeholder carries the dynamic dimension A directly.
    common_flat = tf.placeholder(
        tf.float32, shape=[None, 4], name="common_flat")

    # These placeholders stay within the "one dynamic dimension per leaf" rule.
    domain_lookup = tf.placeholder(
        tf.float32, shape=[None, 4], name="domain_lookup")
    branch_concat_4 = tf.placeholder(
        tf.float32, shape=[None, 240, 24], name="branch_concat_4")
    branch_weights = tf.placeholder(
        tf.float32, shape=[None, 240, 1], name="branch_weights")
    branch_single = tf.placeholder(
        tf.float32, shape=[1, 240, 24], name="branch_single")

    reciprocal = tf.constant(1.0 / 240.0, dtype=tf.float32,
                             name="model/iR_iic/truediv_1_recip")

    # BuildCommonEmbInput path: flat A x 4 -> [A / 240, 240, 4] -> [A / 240, 960]
    common_reshape = tf.reshape(
        common_flat, [-1, 240, 4], name="BuildCommonEmbInput/Reshape")
    common_scale = tf.reshape(
        tf.linspace(1.0, 4.0, 4), [1, 1, 4], name="Reshape")
    common_mul = tf.multiply(
        common_reshape, common_scale, name="BuildCommonEmbInput/mul")
    common_reshape_2 = tf.reshape(
        common_mul, [-1, 960], name="BuildCommonEmbInput/Reshape_2")
    common_reshape_3 = tf.reshape(
        domain_lookup, [-1, 24], name="BuildCommonEmbInput/Reshape_3")

    # Shared Tile multiples: Shape(Reshape_2) -> StridedSlice -> Pack.
    common_shape = tf.shape(common_reshape_2, out_type=tf.int32, name="Shape")
    first_dim = tf.strided_slice(
        common_shape, [0], [1], [1], name="strided_slice_1")
    tile_multiples = tf.stack(
        [tf.squeeze(first_dim, axis=0), 1], axis=0, name="Tile/multiples")

    # Tile-side path, shaped to mirror truediv_1 -> Tile_3.
    truediv_1_input = tf.reduce_sum(
        branch_single, axis=1, name="iR_iic/Sum")
    iR_truediv_1 = tf.multiply(
        reciprocal, truediv_1_input, name="iR_iic/truediv_1")
    tile_3 = tf.tile(iR_truediv_1, tile_multiples, name="Tile_3")

    # truediv_2-side path, shaped to mirror concat_4 -> mul_1 -> Sum_1 -> truediv_2.
    iR_concat_4 = tf.identity(branch_concat_4, name="iR_iic/concat_4")
    iR_mul_1 = tf.multiply(iR_concat_4, branch_weights, name="iR_iic/mul_1")
    iR_sum_1 = tf.reduce_sum(iR_mul_1, axis=1, name="iR_iic/Sum_1")
    iR_truediv_2 = tf.multiply(
        reciprocal, iR_sum_1, name="iR_iic/truediv_2")

    # Two-input version of the big concat, keeping the real op name.
    big_concat = tf.concat(
        [tile_3, iR_truediv_2], axis=1, name="MMoE_input_emb_concat")

  inputs = {
      "common_flat": common_flat,
      "domain_lookup": domain_lookup,
      "branch_concat_4": branch_concat_4,
      "branch_weights": branch_weights,
      "branch_single": branch_single,
  }
  outputs = {
      "buildcommon_reshape_2": common_reshape_2,
      "buildcommon_reshape_3": common_reshape_3,
      "tile_multiples": tile_multiples,
      "iR_truediv_1": iR_truediv_1,
      "tile_3": tile_3,
      "iR_truediv_2": iR_truediv_2,
      "output": big_concat,
  }
  return inputs, outputs


def make_feeds(a_value):
  if a_value % 240 != 0:
    raise ValueError("A must be divisible by 240, got {}".format(a_value))

  groups = a_value // 240

  common_flat = [
      [float(row * 4 + col) for col in range(4)] for row in range(a_value)
  ]
  domain_lookup = [
      [float(row * 24 + col) + 0.5 for col in range(24)]
      for row in range(groups)
  ]
  branch_concat_4 = []
  for group in range(groups):
    group_rows = []
    for row in range(240):
      values = []
      for col in range(24):
        flat_index = ((group * 240 + row) * 24) + col
        values.append(float(flat_index) / 100.0)
      group_rows.append(values)
    branch_concat_4.append(group_rows)

  branch_weights = []
  total_weights = max(groups * 240 - 1, 1)
  for group in range(groups):
    group_rows = []
    for row in range(240):
      flat_index = group * 240 + row
      value = 0.25 + (float(flat_index) / float(total_weights))
      group_rows.append([value])
    branch_weights.append(group_rows)

  branch_single = [[
      [float(row * 24 + col) / 50.0 for col in range(24)]
      for row in range(240)
  ]]

  return {
      "common_flat": common_flat,
      "domain_lookup": domain_lookup,
      "branch_concat_4": branch_concat_4,
      "branch_weights": branch_weights,
      "branch_single": branch_single,
  }


def parse_args():
  parser = argparse.ArgumentParser(
      description=(
          "Build a small SavedModel that mimics the cluster_2 concat pattern "
          "with one Tile input and one truediv input feeding "
          "model/MMoE_input_emb_concat."))
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
      "--run_only",
      action="store_true",
      help="Run the graph once instead of exporting a SavedModel.")
  return parser.parse_args()


def main():
  args = parse_args()
  inputs, outputs = build_graph()
  signature = _build_signature(inputs, outputs)
  feeds = make_feeds(args.a_value)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {inputs[name]: value for name, value in feeds.items()}
    fetched = sess.run(outputs, feed_dict=feed_dict)
    print("Executed graph with A={} -> {}".format(
        args.a_value,
        {name: tuple(value.shape) for name, value in fetched.items()}))

    if not args.run_only:
      _save_model(sess, args.output_dir, signature)
      print("Saved model to {}".format(args.output_dir))


if __name__ == "__main__":
  main()
