import argparse
import os
import shutil

import numpy as np
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()

_BATCH_SIZE = 3
_SEQ_LEN = 240

_COMMON_EMBED_WIDTH = 4
_COMMON_SPLIT_WIDTH = _COMMON_EMBED_WIDTH // 2

_ACT_WIDTH = 4
_ATTR_WIDTH = 20
_CONCAT_WIDTH = _ACT_WIDTH + _ATTR_WIDTH

_MLP_IN_WIDTH = 8
_DE_BEHAVIOR_ZERO_ROWS = 1
_DE_ATTR_ZERO_ROWS = 1


def build_concat_debug_model():
  """Builds a repro that resembles the de_iic/truediv_1 branch."""
  common_flat_a_input = tf.placeholder(
      dtype=tf.float32,
      shape=[None, _COMMON_SPLIT_WIDTH],
      name="common_flat_a")
  common_flat_b_input = tf.placeholder(
      dtype=tf.float32,
      shape=[None, _COMMON_SPLIT_WIDTH],
      name="common_flat_b")
  sequence_mask_input = tf.placeholder(
      dtype=tf.float32,
      shape=[None, _COMMON_EMBED_WIDTH],
      name="sequence_mask")

  de_behavior_flat_input = tf.placeholder(
      dtype=tf.float32,
      shape=[None, _ACT_WIDTH],
      name="de_behavior_flat")
  de_attr_flat_input = tf.placeholder(
      dtype=tf.float32,
      shape=[None, _ATTR_WIDTH],
      name="de_attr_flat")
  mlp_input_input = tf.placeholder(
      dtype=tf.float32,
      shape=[None, _MLP_IN_WIDTH],
      name="mlp_input")

  common_flat_a = tf.reshape(
      common_flat_a_input, [-1, _COMMON_SPLIT_WIDTH], name="common_flat_a_dynamic")
  common_flat_b = tf.reshape(
      common_flat_b_input, [-1, _COMMON_SPLIT_WIDTH], name="common_flat_b_dynamic")
  sequence_mask = tf.reshape(
      sequence_mask_input, [-1, _COMMON_EMBED_WIDTH], name="sequence_mask_dynamic")
  de_behavior_flat = tf.reshape(
      de_behavior_flat_input, [-1, _ACT_WIDTH], name="de_behavior_flat_dynamic")
  de_attr_flat = tf.reshape(
      de_attr_flat_input, [-1, _ATTR_WIDTH], name="de_attr_flat_dynamic")
  mlp_input = tf.reshape(
      mlp_input_input, [-1, _MLP_IN_WIDTH], name="mlp_input_dynamic")

  with tf.name_scope("BuildCommonEmbInput"):
    common_flat_count = tf.shape(common_flat_a, out_type=tf.int32)[0]
    common_batch_size = tf.math.floordiv(
        common_flat_count, _SEQ_LEN, name="common_batch_size")
    sparse_features_concat = tf.concat(
        [common_flat_a, common_flat_b],
        axis=1,
        name="sparse_features_embedding_concat")
    common_shape = tf.stack(
        [common_batch_size, tf.constant(_SEQ_LEN, dtype=tf.int32),
         tf.constant(_COMMON_EMBED_WIDTH, dtype=tf.int32)],
        axis=0,
        name="common_reshape_shape")
    common_reshape = tf.reshape(
        sparse_features_concat, common_shape, name="Reshape")
    sequence_mask_reshape = tf.reshape(
        sequence_mask, common_shape, name="sequence_mask_reshape")
    common_mul = tf.multiply(common_reshape, sequence_mask_reshape, name="mul")
    left_branch = tf.reshape(common_mul, [-1, _SEQ_LEN * _COMMON_EMBED_WIDTH],
                             name="Reshape_2")

  with tf.name_scope("de"):
    de_behavior_zeros = tf.zeros(
        [_DE_BEHAVIOR_ZERO_ROWS, _ACT_WIDTH],
        dtype=tf.float32,
        name="de_behavior_emb_lookup/zeros")
    de_attr_zeros = tf.zeros(
        [_DE_ATTR_ZERO_ROWS, _ATTR_WIDTH],
        dtype=tf.float32,
        name="de_attr_emb_lookup/zeros")

    de_behavior_concat = tf.concat(
        [de_behavior_flat, de_behavior_zeros],
        axis=0,
        name="de_behavior_emb_lookup/concat")
    de_attr_concat = tf.concat(
        [de_attr_flat, de_attr_zeros],
        axis=0,
        name="de_attr_emb_lookup/concat")

    de_behavior_reshape = tf.reshape(
        de_behavior_concat, [-1, _ACT_WIDTH], name="Reshape")
    de_attr_reshape = tf.reshape(
        de_attr_concat, [-1, _ATTR_WIDTH], name="Reshape_1")

  with tf.name_scope("de_iic"):
    zeros = tf.zeros([1, _ACT_WIDTH], dtype=tf.float32, name="zeros")
    zeros_1 = tf.zeros([1, _ATTR_WIDTH], dtype=tf.float32, name="zeros_1")

    concat = tf.concat([zeros, de_behavior_reshape], axis=0, name="concat")
    concat_1 = tf.concat([zeros_1, de_attr_reshape], axis=0, name="concat_1")

    dynamic_flat_count = tf.shape(de_behavior_flat, out_type=tf.int32)[0]
    session_indices = tf.range(
        1, dynamic_flat_count + 1, dtype=tf.int32, name="session_indices")

    act_padding_session = tf.gather(
        concat, session_indices, axis=0, name="act_padding_session")
    attr_padding_session = tf.gather(
        concat_1, session_indices, axis=0, name="attr_padding_session")

    # Mirror the real graph: reshapes are driven by packed shape tensors that
    # reuse the dynamic flattened session length and introduce the sequence
    # width explicitly.
    flat_shape = tf.shape(act_padding_session, out_type=tf.int32, name="flat_shape")
    flat_count = tf.strided_slice(
        flat_shape,
        begin=[0],
        end=[1],
        strides=[1],
        name="strided_slice_1")
    batch_size = tf.math.floordiv(flat_count[0], _SEQ_LEN, name="batch_size")

    reshape_shape = tf.stack(
        [batch_size, tf.constant(_SEQ_LEN, dtype=tf.int32),
         tf.constant(_ACT_WIDTH, dtype=tf.int32)],
        axis=0,
        name="Reshape_2/shape")
    reshape_1_shape = tf.stack(
        [batch_size, tf.constant(_SEQ_LEN, dtype=tf.int32),
         tf.constant(_ATTR_WIDTH, dtype=tf.int32)],
        axis=0,
        name="Reshape_1/shape")

    reshape = tf.reshape(act_padding_session, reshape_shape, name="Reshape")
    reshape_1 = tf.reshape(attr_padding_session, reshape_1_shape, name="Reshape_1")
    concat_4 = tf.concat([reshape, reshape_1], axis=2, name="concat_4")

    mlp_weight = tf.constant(
        np.linspace(
            0.05, 0.95, _MLP_IN_WIDTH * _SEQ_LEN, dtype=np.float32).reshape(
                _MLP_IN_WIDTH, _SEQ_LEN),
        dtype=tf.float32,
        name="de_iic_weight_MLP_dense_0/MatMul/fused_weight")
    mlp_bias = tf.constant(
        np.linspace(-0.2, 0.2, _SEQ_LEN, dtype=np.float32),
        dtype=tf.float32,
        name="de_iic_weight_MLP_dense_0/add/fused_bias")

    mlp_matmul = tf.matmul(
        mlp_input, mlp_weight, name="de_iic_weight_MLP/de_iic_weight_MLP_dense_0/MatMul")
    mlp_add = tf.add(
        mlp_matmul,
        mlp_bias,
        name="de_iic_weight_MLP/de_iic_weight_MLP_dense_0/add")
    mlp_sigmoid = tf.sigmoid(
        mlp_add,
        name="de_iic_weight_MLP/de_iic_weight_MLP_dense_0/sigmoid/Sigmoid")

    expanded = tf.expand_dims(mlp_sigmoid, axis=2, name="ExpandDims")
    slice_end = tf.stack(
        [batch_size, tf.constant(_SEQ_LEN, dtype=tf.int32),
         tf.constant(1, dtype=tf.int32)],
        axis=0,
        name="strided_slice_3/stack_1")
    gating_slice = tf.strided_slice(
        expanded,
        begin=tf.constant([0, 0, 0], dtype=tf.int32),
        end=slice_end,
        strides=tf.constant([1, 1, 1], dtype=tf.int32),
        name="strided_slice_3")

    mul = tf.multiply(concat_4, gating_slice, name="mul")
    sum_1 = tf.reduce_sum(mul, axis=1, keepdims=False, name="Sum_1")

    recip = tf.constant(
        1.0 / float(_SEQ_LEN), dtype=tf.float32, name="truediv_1_recip")
    truediv_1 = tf.multiply(sum_1, recip, name="truediv_1")

  input_emb_concat = tf.concat(
      [left_branch, truediv_1], axis=1, name="input_emb_concat")

  flat_count_value = _BATCH_SIZE * _SEQ_LEN
  feeds = {
      common_flat_a_input: np.arange(
          flat_count_value * _COMMON_SPLIT_WIDTH,
          dtype=np.float32).reshape(flat_count_value, _COMMON_SPLIT_WIDTH),
      common_flat_b_input: (10000 + np.arange(
          flat_count_value * _COMMON_SPLIT_WIDTH,
          dtype=np.float32)).reshape(flat_count_value, _COMMON_SPLIT_WIDTH),
      sequence_mask_input: np.linspace(
          0.25,
          1.25,
          flat_count_value * _COMMON_EMBED_WIDTH,
          dtype=np.float32).reshape(flat_count_value, _COMMON_EMBED_WIDTH),
      de_behavior_flat_input: np.arange(
          flat_count_value * _ACT_WIDTH,
          dtype=np.float32).reshape(flat_count_value, _ACT_WIDTH),
      de_attr_flat_input: (1000 + np.arange(
          flat_count_value * _ATTR_WIDTH,
          dtype=np.float32)).reshape(flat_count_value, _ATTR_WIDTH),
      mlp_input_input: np.linspace(
          -1.0,
          1.0,
          _BATCH_SIZE * _MLP_IN_WIDTH,
          dtype=np.float32).reshape(_BATCH_SIZE, _MLP_IN_WIDTH),
  }
  inputs = {
      "common_flat_a": common_flat_a_input,
      "common_flat_b": common_flat_b_input,
      "sequence_mask": sequence_mask_input,
      "de_behavior_flat": de_behavior_flat_input,
      "de_attr_flat": de_attr_flat_input,
      "mlp_input": mlp_input_input,
  }
  outputs = {
      "de_behavior_reshape": de_behavior_reshape,
      "de_attr_reshape": de_attr_reshape,
      "reshape": reshape,
      "reshape_1": reshape_1,
      "concat_4": concat_4,
      "gating_slice": gating_slice,
      "sum_1": sum_1,
      "truediv_1": truediv_1,
      "input_emb_concat": input_emb_concat,
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
          "Generate a TensorFlow v1 SavedModel for a concat debug subgraph "
          "that more closely mirrors the de_iic/truediv_1 path."))
  parser.add_argument(
      "--save_model",
      action="store_true",
      help="Save the model instead of executing it.")
  parser.add_argument(
      "--export_dir",
      default=os.path.join("model_BESPOKE2", "1"),
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
        print("de_behavior_reshape shape:", fetched["de_behavior_reshape"].shape)
        print("de_attr_reshape shape:", fetched["de_attr_reshape"].shape)
        print("reshape shape:", fetched["reshape"].shape)
        print("reshape_1 shape:", fetched["reshape_1"].shape)
        print("concat_4 shape:", fetched["concat_4"].shape)
        print("gating_slice shape:", fetched["gating_slice"].shape)
        print("sum_1 shape:", fetched["sum_1"].shape)
        print("truediv_1 shape:", fetched["truediv_1"].shape)
        print("input_emb_concat shape:", fetched["input_emb_concat"].shape)

  print("Done")


if __name__ == "__main__":
  main()
