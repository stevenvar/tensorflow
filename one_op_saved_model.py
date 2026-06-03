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
  axis = tf.placeholder(dtype=tf.int32, shape=[], name="axis")
  result = op_fn(x, axis=axis, name=op_name)
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
      axis: np.array(1, dtype=np.int32),
  }
  return {"x": x, "axis": axis}, result, feeds


def _reduce_bool(op_fn, op_name):
  x = tf.placeholder(dtype=tf.bool, shape=[None, 5], name="X")
  axis = tf.placeholder(dtype=tf.int32, shape=[], name="axis")
  result = op_fn(x, axis=axis, name=op_name)
  feeds = {
      x: np.array([[True, True, False, False, True],
                   [True, True, True, True, True]], dtype=np.bool_),
      axis: np.array(1, dtype=np.int32),
  }
  return {"x": x, "axis": axis}, result, feeds


def _arg_reduce(op_fn, op_name):
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  axis = tf.placeholder(dtype=tf.int32, shape=[], name="axis")
  result = op_fn(x, axis=axis, output_type=tf.int32, name=op_name)
  feeds = {
      x: np.array([[1.0, 5.0, 3.0, 2.0, 4.0],
                   [8.0, 6.0, 7.0, 10.0, 9.0]], dtype=np.float32),
      axis: np.array(1, dtype=np.int32),
  }
  return {"x": x, "axis": axis}, result, feeds


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
  shape = tf.placeholder(dtype=tf.int32, shape=[2], name="shape")
  result = tf.reshape(x, shape, name="reshape")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
      shape: np.array([5, 2], dtype=np.int32),
  }
  return {"x": x, "shape": shape}, result, feeds


def _transpose():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  perm = tf.placeholder(dtype=tf.int32, shape=[2], name="perm")
  result = tf.transpose(x, perm=perm, name="transpose")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
      perm: np.array([1, 0], dtype=np.int32),
  }
  return {"x": x, "perm": perm}, result, feeds


def _expand_dims():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  axis = tf.placeholder(dtype=tf.int32, shape=[], name="axis")
  result = tf.expand_dims(x, axis=axis, name="expand_dims")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
      axis: np.array(1, dtype=np.int32),
  }
  return {"x": x, "axis": axis}, result, feeds


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
  axis = tf.placeholder(dtype=tf.int32, shape=[], name="axis")
  result = tf.concat([x, y], axis=axis, name="concat")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
      y: np.array([[6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
      axis: np.array(0, dtype=np.int32),
  }
  return {"x": x, "y": y, "axis": axis}, result, feeds


def _split():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 6], name="X")
  axis = tf.placeholder(dtype=tf.int32, shape=[], name="axis")
  result = tf.split(x, num_or_size_splits=2, axis=axis, name="split")[0]
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32),
      axis: np.array(1, dtype=np.int32),
  }
  return {"x": x, "axis": axis}, result, feeds


def _slice():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  begin = tf.placeholder(dtype=tf.int32, shape=[2], name="begin")
  size = tf.placeholder(dtype=tf.int32, shape=[2], name="size")
  result = tf.slice(x, begin, size, name="slice")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
      begin: np.array([0, 1], dtype=np.int32),
      size: np.array([2, 3], dtype=np.int32),
  }
  return {"x": x, "begin": begin, "size": size}, result, feeds


def _strided_slice():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  begin = tf.placeholder(dtype=tf.int32, shape=[2], name="begin")
  end = tf.placeholder(dtype=tf.int32, shape=[2], name="end")
  strides = tf.placeholder(dtype=tf.int32, shape=[2], name="strides")
  result = tf.strided_slice(x, begin, end, strides, name="strided_slice")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
      begin: np.array([0, 0], dtype=np.int32),
      end: np.array([2, 5], dtype=np.int32),
      strides: np.array([1, 2], dtype=np.int32),
  }
  return {"x": x, "begin": begin, "end": end, "strides": strides}, result, feeds


def _gather():
  x = tf.placeholder(dtype=tf.float32, shape=[5, 5], name="X")
  indices = tf.placeholder(dtype=tf.int32, shape=[None], name="indices")
  result = tf.gather(x, indices, name="gather")
  feeds = {
      x: np.arange(25, dtype=np.float32).reshape(5, 5),
      indices: np.array([0, 2, 4], dtype=np.int32),
  }
  return {"x": x, "indices": indices}, result, feeds


def _gather_nd():
  x = tf.placeholder(dtype=tf.float32, shape=[5, 5], name="X")
  indices = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="indices")
  result = tf.gather_nd(x, indices, name="gather_nd")
  feeds = {
      x: np.arange(25, dtype=np.float32).reshape(5, 5),
      indices: np.array([[0, 0], [2, 3], [4, 4]], dtype=np.int32),
  }
  return {"x": x, "indices": indices}, result, feeds


def _one_hot():
  indices = tf.placeholder(dtype=tf.int32, shape=[None], name="indices")
  depth = tf.placeholder(dtype=tf.int32, shape=[], name="depth")
  result = tf.one_hot(indices, depth, name="one_hot")
  feeds = {
      indices: np.array([0, 2, 4], dtype=np.int32),
      depth: np.array(5, dtype=np.int32),
  }
  return {"indices": indices, "depth": depth}, result, feeds


def _matmul():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 5], name="X")
  y = tf.placeholder(dtype=tf.float32, shape=[5, 5], name="Y")
  result = tf.matmul(x, y, name="matmul")
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
      y: np.eye(5, dtype=np.float32),
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
