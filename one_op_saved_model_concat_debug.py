import argparse
import os
import shutil

import numpy as np
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()

_BATCH_SIZE = 3
_WIDTH = 24


def _float_input():
  return tf.placeholder(tf.float32, shape=[None, _WIDTH], name="input")


def _batch_dim(tensor):
  return tf.shape(tensor, out_type=tf.int32)[0]


def _float_feed(offset=0.0):
  values = np.arange(_BATCH_SIZE * _WIDTH, dtype=np.float32)
  return values.reshape(_BATCH_SIZE, _WIDTH) + offset


def _int_feed(offset=0):
  values = np.arange(_BATCH_SIZE * _WIDTH, dtype=np.int32)
  return values.reshape(_BATCH_SIZE, _WIDTH) + offset


def _positive_int_tensor(x):
  return tf.cast(tf.abs(x), tf.int32) + 1


def _const_row_f32(start=0.0):
  return tf.constant(
      np.linspace(start, start + _WIDTH - 1, _WIDTH, dtype=np.float32)
      .reshape(1, _WIDTH),
      dtype=tf.float32)


def _const_row_i32(start=1):
  return tf.constant(
      np.arange(start, start + _WIDTH, dtype=np.int32).reshape(1, _WIDTH),
      dtype=tf.int32)


def _build_binary_addv2():
  x = _float_input()
  y = tf.add(x, _const_row_f32(1.0), name="AddV2")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_binary_sub():
  x = _float_input()
  y = tf.subtract(x, _const_row_f32(2.0), name="Sub")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_binary_mul():
  x = _float_input()
  y = tf.multiply(x, _const_row_f32(1.0), name="Mul")
  return {"input": x}, {"output": y}, {x: _float_feed(1.0)}


def _build_binary_equal():
  x = _float_input()
  y = tf.equal(x, _const_row_f32(0.0), name="Equal")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_binary_not_equal():
  x = _float_input()
  y = tf.not_equal(x, _const_row_f32(0.0), name="NotEqual")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_binary_greater():
  x = _float_input()
  y = tf.greater(x, _const_row_f32(12.0), name="Greater")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_binary_maximum():
  x = _float_input()
  y = tf.maximum(x, _const_row_f32(8.0), name="Maximum")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_binary_minimum():
  x = _float_input()
  y = tf.minimum(x, _const_row_f32(8.0), name="Minimum")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_binary_floordiv():
  x = _float_input()
  xi = _positive_int_tensor(x)
  y = tf.math.floordiv(xi, _const_row_i32(2), name="FloorDiv")
  return {"input": x}, {"output": y}, {x: _float_feed(1.0)}


def _build_binary_floormod():
  x = _float_input()
  xi = _positive_int_tensor(x)
  y = tf.math.floormod(xi, _const_row_i32(2), name="FloorMod")
  return {"input": x}, {"output": y}, {x: _float_feed(1.0)}


def _build_unary_exp():
  x = _float_input()
  y = tf.exp(x / 16.0, name="Exp")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_unary_rsqrt():
  x = _float_input()
  y = tf.math.rsqrt(tf.abs(x) + 1.0, name="Rsqrt")
  return {"input": x}, {"output": y}, {x: _float_feed(1.0)}


def _build_unary_sigmoid():
  x = _float_input()
  y = tf.sigmoid(x / 16.0, name="Sigmoid")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_unary_round():
  x = _float_input()
  y = tf.round(x / 3.0, name="Round")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_cast():
  x = _float_input()
  y = tf.cast(x, tf.int32, name="Cast")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_concatv2():
  x = _float_input()
  y = tf.concat([x, x], axis=1, name="ConcatV2")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_fill():
  x = _float_input()
  shape = tf.stack([_batch_dim(x), tf.constant(_WIDTH, dtype=tf.int32)],
                   axis=0,
                   name="fill_shape")
  y = tf.fill(shape, tf.constant(7.0, dtype=tf.float32), name="Fill")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_range():
  x = _float_input()
  batch = _batch_dim(x)
  y = tf.range(0, batch * _WIDTH, _WIDTH, dtype=tf.int32, name="Range")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_shape():
  x = _float_input()
  y = tf.shape(x, out_type=tf.int32, name="Shape")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_rank():
  x = _float_input()
  y = tf.rank(x, name="Rank")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_size():
  x = _float_input()
  y = tf.size(x, out_type=tf.int32, name="Size")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_reshape():
  x = _float_input()
  shape = tf.stack([_batch_dim(x), tf.constant(6, dtype=tf.int32),
                    tf.constant(4, dtype=tf.int32)],
                   axis=0,
                   name="reshape_shape")
  y = tf.reshape(x, shape, name="Reshape")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_expanddims():
  x = _float_input()
  y = tf.expand_dims(x, axis=1, name="ExpandDims")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_squeeze():
  x = _float_input()
  expanded = tf.expand_dims(x, axis=1, name="expand_for_squeeze")
  y = tf.squeeze(expanded, axis=[1], name="Squeeze")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_sparse_reshape():
  x = _float_input()
  indices = tf.where(tf.greater(x, 0.0), name="where_indices")
  values = tf.gather_nd(x, indices, name="gather_values")
  sparse = tf.SparseTensor(
      indices=tf.cast(indices, tf.int64),
      values=values,
      dense_shape=tf.cast(tf.shape(x, out_type=tf.int32), tf.int64))
  new_shape = tf.stack([
      tf.cast(_batch_dim(x) * 2, tf.int64),
      tf.constant(12, dtype=tf.int64)
  ], axis=0, name="sparse_new_shape")
  reshaped = tf.sparse.reshape(sparse, new_shape, name="SparseReshape")
  outputs = {
      "indices": reshaped.indices,
      "values": reshaped.values,
      "dense_shape": reshaped.dense_shape,
  }
  return {"input": x}, outputs, {x: _float_feed(1.0)}


def _build_zeroslike():
  x = _float_input()
  y = tf.zeros_like(x, name="ZerosLike")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_pack():
  x = _float_input()
  y = tf.stack([x, x], axis=1, name="Pack")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_transpose():
  x = _float_input()
  reshaped = tf.reshape(
      x,
      tf.stack([_batch_dim(x), tf.constant(6, dtype=tf.int32),
                tf.constant(4, dtype=tf.int32)],
               axis=0),
      name="reshape_before_transpose")
  y = tf.transpose(reshaped, perm=[0, 2, 1], name="Transpose")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_stridedslice():
  x = _float_input()
  y = tf.strided_slice(
      x,
      begin=[0, 0],
      end=[_batch_dim(x), 12],
      strides=[1, 1],
      name="StridedSlice")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_tile():
  x = _float_input()
  y = tf.tile(x, [1, 2], name="Tile")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_select():
  x = _float_input()
  cond = tf.greater(x, 12.0, name="select_cond")
  y = tf.raw_ops.Select(condition=cond, x=x, y=-x, name="Select")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_where():
  x = _float_input()
  cond = tf.greater(x, 12.0, name="where_cond")
  y = tf.where(cond, name="Where")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_reduce_sum():
  x = _float_input()
  reshaped = tf.reshape(
      x,
      tf.stack([_batch_dim(x), tf.constant(6, dtype=tf.int32),
                tf.constant(4, dtype=tf.int32)],
               axis=0))
  y = tf.reduce_sum(reshaped, axis=1, name="Sum")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_reduce_prod():
  x = _float_input()
  reshaped = tf.reshape(
      tf.abs(x) + 1.0,
      tf.stack([_batch_dim(x), tf.constant(6, dtype=tf.int32),
                tf.constant(4, dtype=tf.int32)],
               axis=0))
  y = tf.reduce_prod(reshaped, axis=1, name="Prod")
  return {"input": x}, {"output": y}, {x: _float_feed(1.0)}


def _build_reduce_max():
  x = _float_input()
  reshaped = tf.reshape(
      x,
      tf.stack([_batch_dim(x), tf.constant(6, dtype=tf.int32),
                tf.constant(4, dtype=tf.int32)],
               axis=0))
  y = tf.reduce_max(reshaped, axis=1, name="Max")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_matmul():
  x = _float_input()
  weight = tf.constant(
      np.arange(_WIDTH * 8, dtype=np.float32).reshape(_WIDTH, 8),
      dtype=tf.float32,
      name="matmul_weight")
  y = tf.matmul(x, weight, name="MatMul")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_batch_matmulv2():
  x = _float_input()
  batch = _batch_dim(x)
  lhs = tf.reshape(
      x,
      tf.stack([batch, tf.constant(2, dtype=tf.int32),
                tf.constant(12, dtype=tf.int32)],
               axis=0),
      name="lhs")
  rhs_base = tf.constant(
      np.arange(12 * 5, dtype=np.float32).reshape(1, 12, 5),
      dtype=tf.float32,
      name="rhs_base")
  rhs = tf.tile(rhs_base, [batch, 1, 1], name="rhs")
  y = tf.matmul(lhs, rhs, name="BatchMatMulV2")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_einsum():
  x = _float_input()
  weight = tf.constant(
      np.arange(_WIDTH * 8, dtype=np.float32).reshape(_WIDTH, 8),
      dtype=tf.float32,
      name="einsum_weight")
  y = tf.einsum("bi,ij->bj", x, weight, name="Einsum")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_softmax():
  x = _float_input()
  y = tf.nn.softmax(x, axis=1, name="Softmax")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_gatherv2():
  x = _float_input()
  y = tf.gather(x, [0, 3, 7, 11], axis=1, name="GatherV2")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_gathernd():
  x = _float_input()
  batch = _batch_dim(x)
  row_ids = tf.range(batch, dtype=tf.int32)
  col_ids = tf.math.mod(row_ids * 3, _WIDTH)
  indices = tf.stack([row_ids, col_ids], axis=1, name="gathernd_indices")
  y = tf.gather_nd(x, indices, name="GatherNd")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_scatternd():
  x = _float_input()
  batch = _batch_dim(x)
  indices = tf.expand_dims(tf.range(batch, dtype=tf.int32), axis=1,
                           name="scatter_indices")
  updates = tf.gather(x, [0], axis=1)
  shape = tf.stack([batch], axis=0, name="scatter_shape")
  y = tf.scatter_nd(indices, tf.reshape(updates, [-1]), shape, name="ScatterNd")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_dynamicpartition():
  x = _float_input()
  batch = _batch_dim(x)
  row_partitions = tf.math.mod(tf.range(batch, dtype=tf.int32), 2)
  tiled = tf.tile(tf.expand_dims(row_partitions, axis=1), [1, _WIDTH],
                  name="partition_map")
  parts = tf.dynamic_partition(x, tiled, 2, name="DynamicPartition")
  outputs = {"part0": parts[0], "part1": parts[1]}
  return {"input": x}, outputs, {x: _float_feed()}


def _build_parallel_dynamic_stitch():
  x = _float_input()
  flat = tf.reshape(x, [-1], name="flat")
  size = tf.size(flat, out_type=tf.int32)
  idx0 = tf.range(0, size, 2, dtype=tf.int32, name="idx0")
  idx1 = tf.range(1, size, 2, dtype=tf.int32, name="idx1")
  data0 = tf.gather(flat, idx0, name="data0")
  data1 = tf.gather(flat, idx1, name="data1")
  y = tf.dynamic_stitch([idx0, idx1], [data0, data1],
                        name="ParallelDynamicStitch")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_sparse_segment_mean():
  x = _float_input()
  batch = _batch_dim(x)
  indices = tf.range(batch, dtype=tf.int32, name="indices")
  segment_ids = tf.zeros([batch], dtype=tf.int32, name="segment_ids")
  y = tf.sparse_segment_mean(x, indices, segment_ids, name="SparseSegmentMean")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_sparse_segment_sum():
  x = _float_input()
  batch = _batch_dim(x)
  indices = tf.range(batch, dtype=tf.int32, name="indices")
  segment_ids = tf.zeros([batch], dtype=tf.int32, name="segment_ids")
  y = tf.sparse_segment_sum(x, indices, segment_ids, name="SparseSegmentSum")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_unique():
  x = _float_input()
  values = tf.cast(tf.math.mod(tf.round(x[:, 0]), 7), tf.int32, name="values")
  unique_values, unique_ids = tf.unique(values, name="Unique")
  outputs = {
      "values": unique_values,
      "ids": unique_ids,
  }
  return {"input": x}, outputs, {x: _float_feed()}


def _build_identity():
  x = _float_input()
  y = tf.identity(x, name="Identity")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_print():
  x = _float_input()
  y = tf.Print(x, [tf.shape(x)], message="[BESPOKE] Print shape=", name="Print")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_asstring():
  x = _float_input()
  y = tf.as_string(tf.cast(x[:, 0], tf.int32), name="AsString")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_stringjoin():
  x = _float_input()
  left = tf.as_string(tf.cast(x[:, 0], tf.int32), name="left")
  suffix = tf.fill(tf.shape(left), tf.constant("_suffix"), name="suffix")
  y = tf.string_join([left, suffix], name="StringJoin")
  return {"input": x}, {"output": y}, {x: _float_feed()}


def _build_variable_read():
  x = _float_input()
  bias = tf.get_variable(
      "bias",
      initializer=np.arange(_WIDTH, dtype=np.float32),
      use_resource=True)
  read = tf.raw_ops.ReadVariableOp(
      resource=bias.handle, dtype=tf.float32, name="ReadVariableOp")
  y = tf.add(x, read, name="AddV2")
  outputs = {
      "read": read,
      "output": y,
  }
  return {"input": x}, outputs, {x: _float_feed()}


MODEL_BUILDERS = [
    ("addv2", _build_binary_addv2),
    ("sub", _build_binary_sub),
    ("mul", _build_binary_mul),
    ("equal", _build_binary_equal),
    ("not_equal", _build_binary_not_equal),
    ("greater", _build_binary_greater),
    ("maximum", _build_binary_maximum),
    ("minimum", _build_binary_minimum),
    ("floordiv", _build_binary_floordiv),
    ("floormod", _build_binary_floormod),
    ("exp", _build_unary_exp),
    ("rsqrt", _build_unary_rsqrt),
    ("sigmoid", _build_unary_sigmoid),
    ("round", _build_unary_round),
    ("cast", _build_cast),
    ("concatv2", _build_concatv2),
    ("fill", _build_fill),
    ("range", _build_range),
    ("shape", _build_shape),
    ("rank", _build_rank),
    ("size", _build_size),
    ("reshape", _build_reshape),
    ("expanddims", _build_expanddims),
    ("squeeze", _build_squeeze),
    ("sparse_reshape", _build_sparse_reshape),
    ("zeroslike", _build_zeroslike),
    ("pack", _build_pack),
    ("transpose", _build_transpose),
    ("stridedslice", _build_stridedslice),
    ("tile", _build_tile),
    ("select", _build_select),
    ("where", _build_where),
    ("sum", _build_reduce_sum),
    ("prod", _build_reduce_prod),
    ("max", _build_reduce_max),
    ("matmul", _build_matmul),
    ("batchmatmulv2", _build_batch_matmulv2),
    ("einsum", _build_einsum),
    ("softmax", _build_softmax),
    ("gatherv2", _build_gatherv2),
    ("gathernd", _build_gathernd),
    ("scatternd", _build_scatternd),
    ("dynamicpartition", _build_dynamicpartition),
    ("paralleldynamicstitch", _build_parallel_dynamic_stitch),
    ("sparse_segment_mean", _build_sparse_segment_mean),
    ("sparse_segment_sum", _build_sparse_segment_sum),
    ("unique", _build_unique),
    ("identity", _build_identity),
    ("print", _build_print),
    ("asstring", _build_asstring),
    ("stringjoin", _build_stringjoin),
    ("variable_read", _build_variable_read),
]


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
          "Generate a suite of small TensorFlow v1 SavedModels under "
          "bespoke_tests/, one per op family, all driven by an input with "
          "shape [None, 24]."))
  parser.add_argument(
      "--output_root",
      default="bespoke_tests",
      help="Root directory where the SavedModels will be written.")
  parser.add_argument(
      "--models",
      nargs="*",
      default=None,
      help="Optional subset of model names to build.")
  parser.add_argument(
      "--run_only",
      action="store_true",
      help="Execute each graph once instead of exporting SavedModels.")
  return parser.parse_args()


def main():
  args = parse_args()
  selected = set(args.models) if args.models else None
  built = []

  for model_name, builder_fn in MODEL_BUILDERS:
    if selected and model_name not in selected:
      continue

    with tf.Graph().as_default():
      inputs, outputs, feeds = builder_fn()
      signature = build_signature(inputs, outputs)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        if args.run_only:
          fetched = sess.run(outputs, feed_dict=feeds)
          print("Executed {} with outputs {}".format(
              model_name,
              {key: np.shape(value) for key, value in fetched.items()}))
        else:
          export_dir = os.path.join(args.output_root, "model_{}".format(model_name),
                                    "1")
          save_model(sess, export_dir, signature)
          print("Saved {} to {}".format(model_name, export_dir))
    built.append(model_name)

  print("Built {} bespoke models".format(len(built)))


if __name__ == "__main__":
  main()
