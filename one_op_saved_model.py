import argparse
import os
import shutil

import numpy as np
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()


def _binary_elementwise(op_fn, op_name):
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  y = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="Y")
  result = op_fn(x, y, name=op_name)
  feeds = {
      x: np.array([[11.0, 12.0, 13.0, 14.0, 15.0]], dtype=np.float32),
      y: np.array([[2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32),
  }
  return {"x": x, "y": y}, result, feeds


def _binary_positive(op_fn, op_name):
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  y = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="Y")
  result = op_fn(x, y, name=op_name)
  feeds = {
      x: np.array([[2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32),
      y: np.array([[1.5, 2.0, 2.5, 3.0, 3.5]], dtype=np.float32),
  }
  return {"x": x, "y": y}, result, feeds


def _binary_int(op_fn, op_name):
  x = tf.placeholder(dtype=tf.int32, shape=[None, 5], name="X")
  y = tf.placeholder(dtype=tf.int32, shape=[None, 5], name="Y")
  result = op_fn(x, y, name=op_name)
  feeds = {
      x: np.array([[11, 12, 13, 14, 15]], dtype=np.int32),
      y: np.array([[2, 3, 4, 5, 6]], dtype=np.int32),
  }
  return {"x": x, "y": y}, result, feeds


def _bitwise_binary(op_fn, op_name):
  x = tf.placeholder(dtype=tf.int32, shape=[None, 5], name="X")
  y = tf.placeholder(dtype=tf.int32, shape=[None, 5], name="Y")
  result = op_fn(x, y, name=op_name)
  feeds = {
      x: np.array([[1, 3, 5, 7, 9]], dtype=np.int32),
      y: np.array([[2, 3, 4, 5, 6]], dtype=np.int32),
  }
  return {"x": x, "y": y}, result, feeds


def _compare_elementwise(op_fn, op_name):
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  y = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="Y")
  result = op_fn(x, y, name=op_name)
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
      y: np.array([[3.0, 2.0, 1.0, 4.0, 6.0]], dtype=np.float32),
  }
  return {"x": x, "y": y}, result, feeds


def _logical_binary(op_fn, op_name):
  x = tf.placeholder(dtype=tf.bool, shape=[None, 5], name="X")
  y = tf.placeholder(dtype=tf.bool, shape=[None, 5], name="Y")
  result = op_fn(x, y, name=op_name)
  feeds = {
      x: np.array([[True, True, False, False, True]], dtype=np.bool_),
      y: np.array([[True, False, True, False, False]], dtype=np.bool_),
  }
  return {"x": x, "y": y}, result, feeds


def _unary_elementwise(op_fn, op_name, values=None):
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  result = op_fn(x, name=op_name)
  if values is None:
    values = [[-2.0, -1.0, 0.0, 1.0, 2.0]]
  feeds = {
      x: np.array(values, dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _unary_int(op_fn, op_name):
  x = tf.placeholder(dtype=tf.int32, shape=[None, 5], name="X")
  result = op_fn(x, name=op_name)
  feeds = {
      x: np.array([[-2, -1, 0, 1, 2]], dtype=np.int32),
  }
  return {"x": x}, result, feeds


def _logical_unary(op_fn, op_name):
  x = tf.placeholder(dtype=tf.bool, shape=[None, 5], name="X")
  result = op_fn(x, name=op_name)
  feeds = {
      x: np.array([[True, True, False, False, True]], dtype=np.bool_),
  }
  return {"x": x}, result, feeds


def _reduce(op_fn, op_name):
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  axis = tf.constant(1, dtype=tf.int32, name="axis")
  result = op_fn(x, axis=axis, name=op_name)
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _reduce_bool(op_fn, op_name):
  x = tf.placeholder(dtype=tf.bool, shape=[None, 5], name="X")
  axis = tf.constant(1, dtype=tf.int32, name="axis")
  result = op_fn(x, axis=axis, name=op_name)
  feeds = {
      x: np.array([[True, True, False, False, True],
                   [True, True, True, True, True]], dtype=np.bool_),
  }
  return {"x": x}, result, feeds


def _arg_reduce(op_fn, op_name):
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  axis = tf.constant(1, dtype=tf.int32, name="axis")
  result = op_fn(x, axis=axis, output_type=tf.int32, name=op_name)
  feeds = {
      x: np.array([[1.0, 5.0, 3.0, 2.0, 4.0],
                   [8.0, 6.0, 7.0, 10.0, 9.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _shape_op(op_fn, op_name):
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  result = op_fn(x, name=op_name)
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _cast():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  result = tf.cast(x, tf.int32, name="cast")
  feeds = {
      x: np.array([[1.2, 2.3, 3.4, 4.5, 5.6]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _reshape():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  shape = tf.constant([5, 2], dtype=tf.int32, name="shape")
  result = tf.reshape(x, shape, name="reshape")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _transpose():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  perm = tf.constant([1, 0], dtype=tf.int32, name="perm")
  result = tf.transpose(x, perm=perm, name="transpose")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _expand_dims():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  axis = tf.constant(1, dtype=tf.int32, name="axis")
  result = tf.expand_dims(x, axis=axis, name="expand_dims")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _squeeze():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 1, 5], name="X")
  result = tf.squeeze(x, axis=[1], name="squeeze")
  feeds = {
      x: np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _concat():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  y = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="Y")
  axis = tf.constant(0, dtype=tf.int32, name="axis")
  result = tf.concat([x, y], axis=axis, name="concat")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
      y: np.array([[6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {"x": x, "y": y}, result, feeds


def _split():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 6], name="X")
  axis = tf.constant(1, dtype=tf.int32, name="axis")
  result = tf.split(x, num_or_size_splits=2, axis=axis, name="split")[0]
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _slice():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  begin = tf.constant([0, 1], dtype=tf.int32, name="begin")
  size = tf.constant([2, 3], dtype=tf.int32, name="size")
  result = tf.slice(x, begin, size, name="slice")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _strided_slice():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  begin = tf.constant([0, 0], dtype=tf.int32, name="begin")
  end = tf.constant([2, 5], dtype=tf.int32, name="end")
  strides = tf.constant([1, 2], dtype=tf.int32, name="strides")
  result = tf.strided_slice(x, begin, end, strides, name="strided_slice")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _gather():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  indices = tf.placeholder(dtype=tf.int32, shape=[None], name="indices")
  result = tf.gather(x, indices, name="gather")
  feeds = {
      x: np.arange(10, dtype=np.float32).reshape(2, 5),
      indices: np.array([0, 1], dtype=np.int32),
  }
  return {"x": x, "indices": indices}, result, feeds


def _gather_nd():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  indices = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="indices")
  result = tf.gather_nd(x, indices, name="gather_nd")
  feeds = {
      x: np.arange(10, dtype=np.float32).reshape(2, 5),
      indices: np.array([[0, 0], [1, 4]], dtype=np.int32),
  }
  return {"x": x, "indices": indices}, result, feeds


def _one_hot():
  indices = tf.placeholder(dtype=tf.int32, shape=[None], name="indices")
  depth = tf.constant(5, dtype=tf.int32, name="depth")
  result = tf.one_hot(indices, depth, name="one_hot")
  feeds = {
      indices: np.array([0, 2, 4], dtype=np.int32),
  }
  return {"indices": indices}, result, feeds


def _matmul():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 1, 5], name="X")
  y = tf.placeholder(dtype=tf.float32, shape=[None, 5, 5], name="Y")
  result = tf.matmul(x, y, name="matmul")
  feeds = {
      x: np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32),
      y: np.eye(5, dtype=np.float32).reshape(1, 5, 5),
  }
  return {"x": x, "y": y}, result, feeds


def _batch_matmul():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 2, 3], name="X")
  y = tf.placeholder(dtype=tf.float32, shape=[None, 3, 2], name="Y")
  result = tf.matmul(x, y, name="batch_matmul")
  feeds = {
      x: np.array([[[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]]], dtype=np.float32),
      y: np.array([[[1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]]], dtype=np.float32),
  }
  return {"x": x, "y": y}, result, feeds


def _matrix_diag():
  diagonal = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="diagonal")
  result = tf.linalg.diag(diagonal, name="matrix_diag")
  feeds = {
      diagonal: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
  }
  return {"diagonal": diagonal}, result, feeds


def _matrix_diag_part():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5, 5], name="X")
  result = tf.linalg.diag_part(x, name="matrix_diag_part")
  feeds = {
      x: np.arange(25, dtype=np.float32).reshape(1, 5, 5),
  }
  return {"x": x}, result, feeds


def _fft():
  x = tf.placeholder(dtype=tf.complex64, shape=[None, 8], name="X")
  result = tf.signal.fft(x, name="fft")
  feeds = {
      x: np.array([[complex(i, 0.0) for i in range(8)]], dtype=np.complex64),
  }
  return {"x": x}, result, feeds


def _complex_abs():
  x = tf.placeholder(dtype=tf.complex64, shape=[None, 5], name="X")
  result = tf.abs(x, name="complex_abs")
  feeds = {
      x: np.array([[1 + 1j, 2 + 0j, 0 + 3j, 4 - 4j, 5 + 2j]],
                  dtype=np.complex64),
  }
  return {"x": x}, result, feeds


def _complex_from_real_imag():
  real = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="real")
  imag = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="imag")
  result = tf.complex(real, imag, name="complex")
  feeds = {
      real: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
      imag: np.array([[5.0, 4.0, 3.0, 2.0, 1.0]], dtype=np.float32),
  }
  return {"real": real, "imag": imag}, result, feeds


def _betainc():
  a = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="a")
  b = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="b")
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  result = tf.math.betainc(a, b, x, name="betainc")
  feeds = {
      a: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
      b: np.array([[2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32),
      x: np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
  }
  return {"a": a, "b": b, "x": x}, result, feeds


def _complex_unary(op_fn, op_name):
  x = tf.placeholder(dtype=tf.complex64, shape=[None, 5], name="X")
  result = op_fn(x, name=op_name)
  feeds = {
      x: np.array([[1 + 1j, 2 + 0j, 0 + 3j, 4 - 4j, 5 + 2j]],
                  dtype=np.complex64),
  }
  return {"x": x}, result, feeds


def _pad():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  paddings = tf.constant([[0, 1], [1, 0]], dtype=tf.int32, name="paddings")
  result = tf.pad(x, paddings, name="pad")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _tile():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  multiples = tf.constant([2, 1], dtype=tf.int32, name="multiples")
  result = tf.tile(x, multiples, name="tile")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _reverse():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  axis = tf.constant([1], dtype=tf.int32, name="axis")
  result = tf.reverse(x, axis, name="reverse")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _stack():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  y = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="Y")
  result = tf.stack([x, y], axis=0, name="stack")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
      y: np.array([[6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {"x": x, "y": y}, result, feeds


def _unstack():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 2, 5], name="X")
  result = tf.unstack(x, axis=1, name="unstack")[0]
  feeds = {
      x: np.array([[[1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0]]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _broadcast_to():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="X")
  shape = tf.constant([1, 5], dtype=tf.int32, name="shape")
  result = tf.broadcast_to(x, shape, name="broadcast_to")
  feeds = {
      x: np.array([[1.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _where_indices():
  condition = tf.placeholder(dtype=tf.bool, shape=[None, 5], name="condition")
  result = tf.where(condition, name="where")
  feeds = {
      condition: np.array([[True, False, True, False, True]], dtype=np.bool_),
  }
  return {"condition": condition}, result, feeds


def _where_select():
  condition = tf.placeholder(dtype=tf.bool, shape=[None, 5], name="condition")
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  y = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="Y")
  result = tf.where(condition, x, y, name="where_select")
  feeds = {
      condition: np.array([[True, False, True, False, True]], dtype=np.bool_),
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
      y: np.array([[6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {"condition": condition, "x": x, "y": y}, result, feeds


def _add_n():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  y = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="Y")
  result = tf.add_n([x, y], name="add_n")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
      y: np.array([[6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {"x": x, "y": y}, result, feeds


def _bias_add():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  bias = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0],
                     dtype=tf.float32,
                     name="bias")
  result = tf.nn.bias_add(x, bias, name="bias_add")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _top_k():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  result = tf.nn.top_k(x, k=2, name="top_k").values
  feeds = {
      x: np.array([[1.0, 5.0, 3.0, 2.0, 4.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _pool(op_fn, op_name):
  x = tf.placeholder(dtype=tf.float32, shape=[None, 8, 8, 1], name="X")
  result = op_fn(
      x,
      ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding="VALID",
      name=op_name)
  feeds = {
      x: np.arange(64, dtype=np.float32).reshape(1, 8, 8, 1),
  }
  return {"x": x}, result, feeds


def _lrn():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 3], name="X")
  result = tf.nn.local_response_normalization(x, name="lrn")
  feeds = {
      x: np.arange(48, dtype=np.float32).reshape(1, 4, 4, 3),
  }
  return {"x": x}, result, feeds


def _depth_to_space():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 4], name="X")
  result = tf.nn.depth_to_space(x, block_size=2, name="depth_to_space")
  feeds = {
      x: np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4),
  }
  return {"x": x}, result, feeds


def _space_to_depth():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 1], name="X")
  result = tf.nn.space_to_depth(x, block_size=2, name="space_to_depth")
  feeds = {
      x: np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1),
  }
  return {"x": x}, result, feeds


def _cumulative(op_fn, op_name):
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  result = op_fn(x, axis=1, name=op_name)
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _unsorted_segment(op_fn, op_name):
  data = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="data")
  segment_ids = tf.placeholder(dtype=tf.int32, shape=[None], name="segment_ids")
  result = op_fn(data, segment_ids, num_segments=2, name=op_name)
  feeds = {
      data: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0, 9.0, 10.0],
                      [11.0, 12.0, 13.0, 14.0, 15.0]], dtype=np.float32),
      segment_ids: np.array([0, 1, 0], dtype=np.int32),
  }
  return {"data": data, "segment_ids": segment_ids}, result, feeds


def _dynamic_partition():
  data = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="data")
  partitions = tf.placeholder(dtype=tf.int32, shape=[None], name="partitions")
  result = tf.dynamic_partition(data, partitions, num_partitions=2,
                                name="dynamic_partition")[0]
  feeds = {
      data: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0, 9.0, 10.0],
                      [11.0, 12.0, 13.0, 14.0, 15.0]], dtype=np.float32),
      partitions: np.array([0, 1, 0], dtype=np.int32),
  }
  return {"data": data, "partitions": partitions}, result, feeds


def _dynamic_stitch():
  indices_0 = tf.constant([0], dtype=tf.int32, name="indices_0")
  indices_1 = tf.constant([1], dtype=tf.int32, name="indices_1")
  data_0 = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="data_0")
  data_1 = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="data_1")
  result = tf.dynamic_stitch([indices_0, indices_1], [data_0, data_1],
                             name="dynamic_stitch")
  feeds = {
      data_0: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
      data_1: np.array([[6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
  }
  return {
      "data_0": data_0,
      "data_1": data_1
  }, result, feeds


def _boolean_mask():
  tensor = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="tensor")
  mask = tf.placeholder(dtype=tf.bool, shape=[None], name="mask")
  result = tf.boolean_mask(tensor, mask, name="boolean_mask")
  feeds = {
      tensor: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
      mask: np.array([True, False], dtype=np.bool_),
  }
  return {"tensor": tensor, "mask": mask}, result, feeds


def _sequence_mask():
  lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="lengths")
  result = tf.sequence_mask(lengths, maxlen=5, name="sequence_mask")
  feeds = {
      lengths: np.array([1, 3, 5], dtype=np.int32),
  }
  return {"lengths": lengths}, result, feeds


def _conv2d():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 8, 8, 1], name="X")
  filters = tf.constant(np.ones((3, 3, 1, 2), dtype=np.float32),
                        dtype=tf.float32,
                        name="filters")
  result = tf.nn.conv2d(x, filters, strides=[1, 1, 1, 1], padding="VALID",
                        name="conv2d")
  feeds = {
      x: np.arange(64, dtype=np.float32).reshape(1, 8, 8, 1),
  }
  return {"x": x}, result, feeds


def _depthwise_conv2d():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 8, 8, 1], name="X")
  filters = tf.constant(np.ones((3, 3, 1, 2), dtype=np.float32),
                        dtype=tf.float32,
                        name="filters")
  result = tf.nn.depthwise_conv2d(
      x, filters, strides=[1, 1, 1, 1], padding="VALID",
      name="depthwise_conv2d")
  feeds = {
      x: np.arange(64, dtype=np.float32).reshape(1, 8, 8, 1),
  }
  return {"x": x}, result, feeds


def _conv3d():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 4, 1], name="X")
  filters = tf.constant(np.ones((2, 2, 2, 1, 2), dtype=np.float32),
                        dtype=tf.float32,
                        name="filters")
  result = tf.nn.conv3d(x, filters, strides=[1, 1, 1, 1, 1],
                        padding="VALID", name="conv3d")
  feeds = {
      x: np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4, 1),
  }
  return {"x": x}, result, feeds


def _batch_to_space():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 2, 2, 1], name="X")
  result = tf.batch_to_space_nd(
      x, block_shape=[2, 2], crops=[[0, 0], [0, 0]],
      name="batch_to_space")
  feeds = {
      x: np.arange(16, dtype=np.float32).reshape(4, 2, 2, 1),
  }
  return {"x": x}, result, feeds


def _space_to_batch():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 1], name="X")
  result = tf.space_to_batch_nd(
      x, block_shape=[2, 2], paddings=[[0, 0], [0, 0]],
      name="space_to_batch")
  feeds = {
      x: np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1),
  }
  return {"x": x}, result, feeds


def _resize(op_fn, op_name):
  images = tf.placeholder(dtype=tf.float32, shape=[None, 8, 8, 1],
                          name="images")
  result = op_fn(images, size=[4, 4], name=op_name)
  feeds = {
      images: np.arange(64, dtype=np.float32).reshape(1, 8, 8, 1),
  }
  return {"images": images}, result, feeds


def _extract_image_patches():
  images = tf.placeholder(dtype=tf.float32, shape=[None, 8, 8, 1],
                          name="images")
  result = tf.image.extract_image_patches(
      images, ksizes=[1, 2, 2, 1], strides=[1, 2, 2, 1],
      rates=[1, 1, 1, 1], padding="VALID", name="extract_image_patches")
  feeds = {
      images: np.arange(64, dtype=np.float32).reshape(1, 8, 8, 1),
  }
  return {"images": images}, result, feeds


def _extract_volume_patches():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 4, 1], name="X")
  result = tf.extract_volume_patches(
      x, ksizes=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
      padding="VALID", name="extract_volume_patches")
  feeds = {
      x: np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4, 1),
  }
  return {"x": x}, result, feeds


def _crop_and_resize():
  image = tf.placeholder(dtype=tf.float32, shape=[None, 8, 8, 1], name="image")
  boxes = tf.placeholder(dtype=tf.float32, shape=[None, 4], name="boxes")
  box_ind = tf.placeholder(dtype=tf.int32, shape=[None], name="box_ind")
  result = tf.image.crop_and_resize(image, boxes, box_ind, crop_size=[4, 4],
                                    name="crop_and_resize")
  feeds = {
      image: np.arange(64, dtype=np.float32).reshape(1, 8, 8, 1),
      boxes: np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
      box_ind: np.array([0], dtype=np.int32),
  }
  return {"image": image, "boxes": boxes, "box_ind": box_ind}, result, feeds


def _unique():
  x = tf.placeholder(dtype=tf.int32, shape=[None], name="X")
  result = tf.unique(x, name="unique").y
  feeds = {
      x: np.array([1, 2, 1, 3, 2], dtype=np.int32),
  }
  return {"x": x}, result, feeds


def _unique_with_counts():
  x = tf.placeholder(dtype=tf.int32, shape=[None], name="X")
  result = tf.unique_with_counts(x, name="unique_with_counts").y
  feeds = {
      x: np.array([1, 2, 1, 3, 2], dtype=np.int32),
  }
  return {"x": x}, result, feeds


def _in_top_k():
  predictions = tf.placeholder(dtype=tf.float32, shape=[None, 5],
                               name="predictions")
  targets = tf.placeholder(dtype=tf.int32, shape=[None], name="targets")
  result = tf.nn.in_top_k(predictions, targets, k=2, name="in_top_k")
  feeds = {
      predictions: np.array([[0.1, 0.2, 0.9, 0.3, 0.4],
                             [0.8, 0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
      targets: np.array([2, 0], dtype=np.int32),
  }
  return {"predictions": predictions, "targets": targets}, result, feeds


def _softmax_cross_entropy():
  labels = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="labels")
  logits = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="logits")
  result = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=labels, logits=logits, name="softmax_cross_entropy")
  feeds = {
      labels: np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
      logits: np.array([[0.1, 0.2, 0.9, 0.3, 0.4]], dtype=np.float32),
  }
  return {"labels": labels, "logits": logits}, result, feeds


def _spd_batch():
  return np.array([[[4.0, 1.0, 1.0],
                    [1.0, 4.0, 1.0],
                    [1.0, 1.0, 4.0]]], dtype=np.float32)


def _matrix_inverse():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3], name="X")
  result = tf.linalg.inv(x, name="matrix_inverse")
  feeds = {
      x: _spd_batch(),
  }
  return {"x": x}, result, feeds


def _matrix_solve():
  matrix = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3], name="matrix")
  rhs = tf.placeholder(dtype=tf.float32, shape=[None, 3, 1], name="rhs")
  result = tf.linalg.solve(matrix, rhs, name="matrix_solve")
  feeds = {
      matrix: _spd_batch(),
      rhs: np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32),
  }
  return {"matrix": matrix, "rhs": rhs}, result, feeds


def _cholesky():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3], name="X")
  result = tf.linalg.cholesky(x, name="cholesky")
  feeds = {
      x: _spd_batch(),
  }
  return {"x": x}, result, feeds


def _qr():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3], name="X")
  result = tf.linalg.qr(x, full_matrices=False, name="qr").q
  feeds = {
      x: np.arange(9, dtype=np.float32).reshape(1, 3, 3) + 1.0,
  }
  return {"x": x}, result, feeds


def _svd():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3], name="X")
  result = tf.linalg.svd(x, compute_uv=True, full_matrices=False,
                         name="svd").s
  feeds = {
      x: np.arange(9, dtype=np.float32).reshape(1, 3, 3) + 1.0,
  }
  return {"x": x}, result, feeds


def _triangular_solve():
  matrix = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3], name="matrix")
  rhs = tf.placeholder(dtype=tf.float32, shape=[None, 3, 1], name="rhs")
  result = tf.linalg.triangular_solve(matrix, rhs, lower=True,
                                      name="triangular_solve")
  feeds = {
      matrix: np.array([[[2.0, 0.0, 0.0],
                         [1.0, 3.0, 0.0],
                         [1.0, 1.0, 4.0]]], dtype=np.float32),
      rhs: np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32),
  }
  return {"matrix": matrix, "rhs": rhs}, result, feeds


def _matrix_band_part():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4], name="X")
  result = tf.linalg.band_part(x, num_lower=1, num_upper=1,
                               name="matrix_band_part")
  feeds = {
      x: np.arange(16, dtype=np.float32).reshape(1, 4, 4),
  }
  return {"x": x}, result, feeds


def _matrix_set_diag():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4], name="X")
  diagonal = tf.placeholder(dtype=tf.float32, shape=[None, 4], name="diagonal")
  result = tf.linalg.set_diag(x, diagonal, name="matrix_set_diag")
  feeds = {
      x: np.zeros((1, 4, 4), dtype=np.float32),
      diagonal: np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
  }
  return {"x": x, "diagonal": diagonal}, result, feeds


def _einsum():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 2, 3], name="X")
  y = tf.placeholder(dtype=tf.float32, shape=[None, 3, 2], name="Y")
  result = tf.einsum("bij,bjk->bik", x, y, name="einsum")
  feeds = {
      x: np.arange(6, dtype=np.float32).reshape(1, 2, 3),
      y: np.arange(6, dtype=np.float32).reshape(1, 3, 2),
  }
  return {"x": x, "y": y}, result, feeds


def _matrix_determinant():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3], name="X")
  result = tf.linalg.det(x, name="matrix_determinant")
  feeds = {
      x: _spd_batch(),
  }
  return {"x": x}, result, feeds


def _matrix_trace():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3], name="X")
  result = tf.linalg.trace(x, name="matrix_trace")
  feeds = {
      x: _spd_batch(),
  }
  return {"x": x}, result, feeds


def _scatter_nd():
  indices = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="indices")
  updates = tf.placeholder(dtype=tf.float32, shape=[None], name="updates")
  result = tf.scatter_nd(indices, updates, shape=[2, 5], name="scatter_nd")
  feeds = {
      indices: np.array([[0, 0], [0, 3], [1, 4]], dtype=np.int32),
      updates: np.array([1.0, 2.0, 3.0], dtype=np.float32),
  }
  return {"indices": indices, "updates": updates}, result, feeds


def _searchsorted():
  sorted_sequence = tf.placeholder(dtype=tf.float32, shape=[None, 5],
                                   name="sorted_sequence")
  values = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="values")
  result = tf.searchsorted(sorted_sequence, values, name="searchsorted")
  feeds = {
      sorted_sequence: np.array([[1.0, 3.0, 5.0, 7.0, 9.0]],
                                dtype=np.float32),
      values: np.array([[0.0, 4.0, 10.0]], dtype=np.float32),
  }
  return {"sorted_sequence": sorted_sequence, "values": values}, result, feeds


def _listdiff():
  x = tf.placeholder(dtype=tf.int32, shape=[None], name="X")
  y = tf.placeholder(dtype=tf.int32, shape=[None], name="Y")
  result = tf.setdiff1d(x, y, name="listdiff").out
  feeds = {
      x: np.array([1, 2, 3, 4, 5], dtype=np.int32),
      y: np.array([2, 4], dtype=np.int32),
  }
  return {"x": x, "y": y}, result, feeds


def _image_unary(op_fn, op_name):
  images = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 3],
                          name="images")
  result = op_fn(images, name=op_name)
  feeds = {
      images: np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(1, 4, 4, 3),
  }
  return {"images": images}, result, feeds


def _adjust_contrast():
  images = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 3],
                          name="images")
  result = tf.image.adjust_contrast(images, contrast_factor=1.5,
                                    name="adjust_contrast")
  feeds = {
      images: np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(1, 4, 4, 3),
  }
  return {"images": images}, result, feeds


def _adjust_hue():
  images = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 3],
                          name="images")
  result = tf.image.adjust_hue(images, delta=0.1, name="adjust_hue")
  feeds = {
      images: np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(1, 4, 4, 3),
  }
  return {"images": images}, result, feeds


def _adjust_saturation():
  images = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 3],
                          name="images")
  result = tf.image.adjust_saturation(images, saturation_factor=1.5,
                                      name="adjust_saturation")
  feeds = {
      images: np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(1, 4, 4, 3),
  }
  return {"images": images}, result, feeds


def _non_max_suppression():
  boxes = tf.placeholder(dtype=tf.float32, shape=[None, 4], name="boxes")
  scores = tf.placeholder(dtype=tf.float32, shape=[None], name="scores")
  result = tf.image.non_max_suppression(boxes, scores, max_output_size=2,
                                        name="non_max_suppression")
  feeds = {
      boxes: np.array([[0.0, 0.0, 1.0, 1.0],
                       [0.1, 0.1, 1.1, 1.1],
                       [2.0, 2.0, 3.0, 3.0]], dtype=np.float32),
      scores: np.array([0.9, 0.8, 0.7], dtype=np.float32),
  }
  return {"boxes": boxes, "scores": scores}, result, feeds


def _fake_quant_args():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  result = tf.quantization.fake_quant_with_min_max_args(
      x, min=-6.0, max=6.0, name="fake_quant_with_min_max_args")
  feeds = {
      x: np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _fake_quant_vars():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  min_value = tf.constant(-6.0, dtype=tf.float32, name="min")
  max_value = tf.constant(6.0, dtype=tf.float32, name="max")
  result = tf.quantization.fake_quant_with_min_max_vars(
      x, min_value, max_value, name="fake_quant_with_min_max_vars")
  feeds = {
      x: np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _bitcast():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  result = tf.bitcast(x, tf.int32, name="bitcast")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _saturate_cast():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  result = tf.saturate_cast(x, tf.int8, name="saturate_cast")
  feeds = {
      x: np.array([[-200.0, -2.0, 0.0, 2.0, 200.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


def _string_length():
  x = tf.placeholder(dtype=tf.string, shape=[None], name="X")
  result = tf.strings.length(x, name="string_length")
  feeds = {
      x: np.array([b"alpha", b"beta"], dtype=np.object_),
  }
  return {"x": x}, result, feeds


def _string_join():
  x = tf.placeholder(dtype=tf.string, shape=[None], name="X")
  y = tf.placeholder(dtype=tf.string, shape=[None], name="Y")
  result = tf.strings.join([x, y], separator=b"_", name="string_join")
  feeds = {
      x: np.array([b"alpha", b"beta"], dtype=np.object_),
      y: np.array([b"one", b"two"], dtype=np.object_),
  }
  return {"x": x, "y": y}, result, feeds


def _regex_replace():
  x = tf.placeholder(dtype=tf.string, shape=[None], name="X")
  result = tf.strings.regex_replace(x, pattern="a", rewrite="A",
                                    name="regex_replace")
  feeds = {
      x: np.array([b"alpha", b"beta"], dtype=np.object_),
  }
  return {"x": x}, result, feeds


def _substr():
  x = tf.placeholder(dtype=tf.string, shape=[None], name="X")
  result = tf.strings.substr(x, pos=1, len=2, name="substr")
  feeds = {
      x: np.array([b"alpha", b"beta"], dtype=np.object_),
  }
  return {"x": x}, result, feeds


def _as_string():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  result = tf.as_string(x, name="as_string")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
  }
  return {"x": x}, result, feeds


OPS = {
    0: ("add", lambda: _binary_elementwise(tf.add, "add")),
    1: ("sub", lambda: _binary_elementwise(tf.subtract, "sub")),
    2: ("mul", lambda: _binary_elementwise(tf.multiply, "mul")),
    3: ("div", lambda: _binary_elementwise(tf.divide, "div")),
    4: ("maximum", lambda: _binary_elementwise(tf.maximum, "maximum")),
    5: ("minimum", lambda: _binary_elementwise(tf.minimum, "minimum")),
    6: ("squared_difference",
        lambda: _binary_elementwise(tf.squared_difference,
                                    "squared_difference")),
    7: ("pow", lambda: _binary_elementwise(tf.pow, "pow")),
    8: ("neg", lambda: _unary_elementwise(tf.negative, "neg")),
    9: ("matmul", _matmul),
    10: ("real_div", lambda: _binary_positive(tf.realdiv, "real_div")),
    11: ("floor_div", lambda: _binary_int(tf.floor_div, "floor_div")),
    12: ("floor_mod", lambda: _binary_int(tf.floormod, "floor_mod")),
    13: ("abs", lambda: _unary_elementwise(tf.abs, "abs")),
    14: ("sign", lambda: _unary_elementwise(tf.sign, "sign")),
    15: ("square", lambda: _unary_elementwise(tf.square, "square")),
    16: ("sqrt",
         lambda: _unary_elementwise(tf.sqrt, "sqrt",
                                    [[1.0, 4.0, 9.0, 16.0, 25.0]])),
    17: ("rsqrt",
         lambda: _unary_elementwise(tf.rsqrt, "rsqrt",
                                    [[1.0, 4.0, 9.0, 16.0, 25.0]])),
    18: ("exp", lambda: _unary_elementwise(tf.exp, "exp")),
    19: ("expm1", lambda: _unary_elementwise(tf.expm1, "expm1")),
    20: ("log",
         lambda: _unary_elementwise(tf.log, "log",
                                    [[1.0, 2.0, 3.0, 4.0, 5.0]])),
    21: ("log1p",
         lambda: _unary_elementwise(tf.log1p, "log1p",
                                    [[0.0, 1.0, 2.0, 3.0, 4.0]])),
    22: ("sin", lambda: _unary_elementwise(tf.sin, "sin")),
    23: ("cos", lambda: _unary_elementwise(tf.cos, "cos")),
    24: ("tan", lambda: _unary_elementwise(tf.tan, "tan")),
    25: ("asin",
         lambda: _unary_elementwise(tf.asin, "asin",
                                    [[-0.5, -0.25, 0.0, 0.25, 0.5]])),
    26: ("acos",
         lambda: _unary_elementwise(tf.acos, "acos",
                                    [[-0.5, -0.25, 0.0, 0.25, 0.5]])),
    27: ("atan", lambda: _unary_elementwise(tf.atan, "atan")),
    28: ("sinh", lambda: _unary_elementwise(tf.sinh, "sinh")),
    29: ("cosh", lambda: _unary_elementwise(tf.cosh, "cosh")),
    30: ("tanh", lambda: _unary_elementwise(tf.tanh, "tanh")),
    31: ("asinh", lambda: _unary_elementwise(tf.asinh, "asinh")),
    32: ("acosh",
         lambda: _unary_elementwise(tf.acosh, "acosh",
                                    [[1.0, 1.25, 1.5, 2.0, 3.0]])),
    33: ("atanh",
         lambda: _unary_elementwise(tf.atanh, "atanh",
                                    [[-0.5, -0.25, 0.0, 0.25, 0.5]])),
    34: ("erf", lambda: _unary_elementwise(tf.erf, "erf")),
    35: ("erfc", lambda: _unary_elementwise(tf.erfc, "erfc")),
    36: ("ceil", lambda: _unary_elementwise(tf.ceil, "ceil")),
    37: ("floor", lambda: _unary_elementwise(tf.floor, "floor")),
    38: ("round", lambda: _unary_elementwise(tf.round, "round")),
    39: ("rint", lambda: _unary_elementwise(tf.rint, "rint")),
    40: ("relu", lambda: _unary_elementwise(tf.nn.relu, "relu")),
    41: ("relu6", lambda: _unary_elementwise(tf.nn.relu6, "relu6")),
    42: ("elu", lambda: _unary_elementwise(tf.nn.elu, "elu")),
    43: ("selu", lambda: _unary_elementwise(tf.nn.selu, "selu")),
    44: ("softplus", lambda: _unary_elementwise(tf.nn.softplus, "softplus")),
    45: ("softsign", lambda: _unary_elementwise(tf.nn.softsign, "softsign")),
    46: ("sigmoid", lambda: _unary_elementwise(tf.sigmoid, "sigmoid")),
    47: ("softmax", lambda: _unary_elementwise(tf.nn.softmax, "softmax")),
    48: ("l2_loss", lambda: _unary_elementwise(tf.nn.l2_loss, "l2_loss")),
    49: ("zeros_like", lambda: _unary_elementwise(tf.zeros_like, "zeros_like")),
    50: ("ones_like", lambda: _unary_elementwise(tf.ones_like, "ones_like")),
    51: ("identity", lambda: _unary_elementwise(tf.identity, "identity")),
    52: ("stop_gradient",
         lambda: _unary_elementwise(tf.stop_gradient, "stop_gradient")),
    53: ("equal", lambda: _compare_elementwise(tf.equal, "equal")),
    54: ("not_equal", lambda: _compare_elementwise(tf.not_equal, "not_equal")),
    55: ("less", lambda: _compare_elementwise(tf.less, "less")),
    56: ("less_equal",
         lambda: _compare_elementwise(tf.less_equal, "less_equal")),
    57: ("greater", lambda: _compare_elementwise(tf.greater, "greater")),
    58: ("greater_equal",
         lambda: _compare_elementwise(tf.greater_equal, "greater_equal")),
    59: ("logical_and", lambda: _logical_binary(tf.logical_and, "logical_and")),
    60: ("logical_or", lambda: _logical_binary(tf.logical_or, "logical_or")),
    61: ("logical_xor", lambda: _logical_binary(tf.logical_xor, "logical_xor")),
    62: ("logical_not", lambda: _logical_unary(tf.logical_not, "logical_not")),
    63: ("bitwise_and",
         lambda: _bitwise_binary(tf.bitwise.bitwise_and, "bitwise_and")),
    64: ("bitwise_or",
         lambda: _bitwise_binary(tf.bitwise.bitwise_or, "bitwise_or")),
    65: ("bitwise_xor",
         lambda: _bitwise_binary(tf.bitwise.bitwise_xor, "bitwise_xor")),
    66: ("invert", lambda: _unary_int(tf.bitwise.invert, "invert")),
    67: ("left_shift",
         lambda: _bitwise_binary(tf.bitwise.left_shift, "left_shift")),
    68: ("right_shift",
         lambda: _bitwise_binary(tf.bitwise.right_shift, "right_shift")),
    69: ("reduce_sum", lambda: _reduce(tf.reduce_sum, "reduce_sum")),
    70: ("reduce_prod", lambda: _reduce(tf.reduce_prod, "reduce_prod")),
    71: ("reduce_mean", lambda: _reduce(tf.reduce_mean, "reduce_mean")),
    72: ("reduce_min", lambda: _reduce(tf.reduce_min, "reduce_min")),
    73: ("reduce_max", lambda: _reduce(tf.reduce_max, "reduce_max")),
    74: ("reduce_any", lambda: _reduce_bool(tf.reduce_any, "reduce_any")),
    75: ("reduce_all", lambda: _reduce_bool(tf.reduce_all, "reduce_all")),
    76: ("argmax", lambda: _arg_reduce(tf.argmax, "argmax")),
    77: ("argmin", lambda: _arg_reduce(tf.argmin, "argmin")),
    78: ("shape", lambda: _shape_op(tf.shape, "shape")),
    79: ("rank", lambda: _shape_op(tf.rank, "rank")),
    80: ("size", lambda: _shape_op(tf.size, "size")),
    81: ("cast", _cast),
    82: ("reshape", _reshape),
    83: ("transpose", _transpose),
    84: ("expand_dims", _expand_dims),
    85: ("squeeze", _squeeze),
    86: ("concat", _concat),
    87: ("split", _split),
    88: ("slice", _slice),
    89: ("strided_slice", _strided_slice),
    90: ("gather", _gather),
    91: ("gather_nd", _gather_nd),
    92: ("one_hot", _one_hot),
    93: ("batch_matmul", _batch_matmul),
    94: ("matrix_diag", _matrix_diag),
    95: ("matrix_diag_part", _matrix_diag_part),
    96: ("fft", _fft),
    97: ("complex_abs", _complex_abs),
    98: ("is_finite", lambda: _unary_elementwise(tf.is_finite, "is_finite")),
    99: ("is_nan", lambda: _unary_elementwise(tf.is_nan, "is_nan")),
    100: ("reciprocal",
          lambda: _unary_elementwise(tf.reciprocal, "reciprocal",
                                     [[1.0, 2.0, 3.0, 4.0, 5.0]])),
    101: ("lgamma",
          lambda: _unary_elementwise(tf.lgamma, "lgamma",
                                     [[1.0, 2.0, 3.0, 4.0, 5.0]])),
    102: ("digamma",
          lambda: _unary_elementwise(tf.digamma, "digamma",
                                     [[1.0, 2.0, 3.0, 4.0, 5.0]])),
    103: ("polygamma",
          lambda: _binary_positive(tf.polygamma, "polygamma")),
    104: ("atan2", lambda: _binary_elementwise(tf.atan2, "atan2")),
    105: ("xdivy", lambda: _binary_positive(tf.math.xdivy, "xdivy")),
    106: ("xlogy", lambda: _binary_positive(tf.math.xlogy, "xlogy")),
    107: ("xlog1py", lambda: _binary_positive(tf.math.xlog1py, "xlog1py")),
    108: ("betainc", _betainc),
    109: ("igamma", lambda: _binary_positive(tf.igamma, "igamma")),
    110: ("igammac", lambda: _binary_positive(tf.igammac, "igammac")),
    111: ("nextafter", lambda: _binary_positive(tf.math.nextafter,
                                                "nextafter")),
    112: ("complex", _complex_from_real_imag),
    113: ("real", lambda: _complex_unary(tf.math.real, "real")),
    114: ("imag", lambda: _complex_unary(tf.math.imag, "imag")),
    115: ("angle", lambda: _complex_unary(tf.math.angle, "angle")),
    116: ("conj", lambda: _complex_unary(tf.math.conj, "conj")),
    117: ("pad", _pad),
    118: ("tile", _tile),
    119: ("reverse", _reverse),
    120: ("stack", _stack),
    121: ("unstack", _unstack),
    122: ("broadcast_to", _broadcast_to),
    123: ("where", _where_indices),
    124: ("where_select", _where_select),
    125: ("add_n", _add_n),
    126: ("bias_add", _bias_add),
    127: ("leaky_relu", lambda: _unary_elementwise(tf.nn.leaky_relu,
                                                  "leaky_relu")),
    128: ("log_softmax", lambda: _unary_elementwise(tf.nn.log_softmax,
                                                    "log_softmax")),
    129: ("top_k", _top_k),
    130: ("avg_pool", lambda: _pool(tf.nn.avg_pool, "avg_pool")),
    131: ("max_pool", lambda: _pool(tf.nn.max_pool, "max_pool")),
    132: ("lrn", _lrn),
    133: ("depth_to_space", _depth_to_space),
    134: ("space_to_depth", _space_to_depth),
    135: ("cumsum", lambda: _cumulative(tf.cumsum, "cumsum")),
    136: ("cumprod", lambda: _cumulative(tf.cumprod, "cumprod")),
    137: ("reduce_logsumexp",
          lambda: _reduce(tf.reduce_logsumexp, "reduce_logsumexp")),
    138: ("unsorted_segment_sum",
          lambda: _unsorted_segment(tf.unsorted_segment_sum,
                                    "unsorted_segment_sum")),
    139: ("unsorted_segment_max",
          lambda: _unsorted_segment(tf.unsorted_segment_max,
                                    "unsorted_segment_max")),
    140: ("unsorted_segment_min",
          lambda: _unsorted_segment(tf.unsorted_segment_min,
                                    "unsorted_segment_min")),
    141: ("dynamic_partition", _dynamic_partition),
    142: ("dynamic_stitch", _dynamic_stitch),
    143: ("boolean_mask", _boolean_mask),
    144: ("sequence_mask", _sequence_mask),
    145: ("conv2d", _conv2d),
    146: ("depthwise_conv2d", _depthwise_conv2d),
    147: ("conv3d", _conv3d),
    148: ("batch_to_space", _batch_to_space),
    149: ("space_to_batch", _space_to_batch),
    150: ("resize_bilinear",
          lambda: _resize(tf.image.resize_bilinear, "resize_bilinear")),
    151: ("resize_nearest_neighbor",
          lambda: _resize(tf.image.resize_nearest_neighbor,
                          "resize_nearest_neighbor")),
    152: ("extract_image_patches", _extract_image_patches),
    153: ("extract_volume_patches", _extract_volume_patches),
    154: ("crop_and_resize", _crop_and_resize),
    155: ("unique", _unique),
    156: ("unique_with_counts", _unique_with_counts),
    157: ("in_top_k", _in_top_k),
    158: ("softmax_cross_entropy", _softmax_cross_entropy),
    159: ("matrix_inverse", _matrix_inverse),
    160: ("matrix_solve", _matrix_solve),
    161: ("cholesky", _cholesky),
    162: ("qr", _qr),
    163: ("svd", _svd),
    164: ("triangular_solve", _triangular_solve),
    165: ("matrix_band_part", _matrix_band_part),
    166: ("matrix_set_diag", _matrix_set_diag),
    167: ("einsum", _einsum),
    168: ("matrix_determinant", _matrix_determinant),
    169: ("matrix_trace", _matrix_trace),
    170: ("scatter_nd", _scatter_nd),
    171: ("searchsorted", _searchsorted),
    172: ("listdiff", _listdiff),
    173: ("rgb_to_hsv", lambda: _image_unary(tf.image.rgb_to_hsv,
                                            "rgb_to_hsv")),
    174: ("hsv_to_rgb", lambda: _image_unary(tf.image.hsv_to_rgb,
                                            "hsv_to_rgb")),
    175: ("adjust_contrast", _adjust_contrast),
    176: ("adjust_hue", _adjust_hue),
    177: ("adjust_saturation", _adjust_saturation),
    178: ("non_max_suppression", _non_max_suppression),
    179: ("fake_quant_with_min_max_args", _fake_quant_args),
    180: ("fake_quant_with_min_max_vars", _fake_quant_vars),
    181: ("bitcast", _bitcast),
    182: ("saturate_cast", _saturate_cast),
    183: ("string_length", _string_length),
    184: ("string_join", _string_join),
    185: ("regex_replace", _regex_replace),
    186: ("substr", _substr),
    187: ("as_string", _as_string),
}


def _model_dir(export_base_dir, op_name):
  return os.path.join(export_base_dir, "model_{}".format(op_name.upper()))


def _build_signature(inputs, result):
  tensor_inputs = {
      key: tf.saved_model.utils.build_tensor_info(tensor)
      for key, tensor in inputs.items()
  }
  tensor_outputs = {
      "outputs": tf.saved_model.utils.build_tensor_info(result),
  }
  return tf.saved_model.signature_def_utils.build_signature_def(
      inputs=tensor_inputs,
      outputs=tensor_outputs,
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


def _parse_args():
  parser = argparse.ArgumentParser(
      description="Generate or run a one-operation TensorFlow v1 SavedModel.")
  parser.add_argument(
      "--op_id",
      type=int,
      default=0,
      choices=sorted(OPS.keys()),
      help="Operation id to generate. Ignored when --all_ops is set.")
  parser.add_argument(
      "--all_ops",
      action="store_true",
      help="Generate or run every available operation.")
  parser.add_argument(
      "--save_model",
      action="store_true",
      help="Save the selected one-operation model instead of executing it.")
  parser.add_argument(
      "--export_base_dir",
      default="unit_tests",
      help="Base directory for SavedModel exports.")
  parser.add_argument(
      "--list_ops",
      action="store_true",
      help="List available operation ids and exit.")
  return parser.parse_args()


def _run_op(op_id, args):
  op_name, op_builder = OPS[op_id]
  export_dir = os.path.join(_model_dir(args.export_base_dir, op_name), "1")

  with tf.Graph().as_default():
    inputs, result, feeds = op_builder()
    signature = _build_signature(inputs, result)

    with tf.Session() as sess:
      if args.save_model:
        _save_model(sess, export_dir, signature)
        print("Saved {} model to {}".format(op_name, export_dir))
      else:
        output = sess.run(result, feed_dict=feeds)
        print("{} output: {}".format(op_name, output))


def main():
  args = _parse_args()

  if args.list_ops:
    for op_id, (op_name, _) in sorted(OPS.items()):
      print("{}: {}".format(op_id, op_name))
    return

  op_ids = sorted(OPS.keys()) if args.all_ops else [args.op_id]
  for op_id in op_ids:
    _run_op(op_id, args)

  print("Done")


if __name__ == "__main__":
  main()
