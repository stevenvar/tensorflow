import argparse
import inspect
import json
import os
import re
import shutil

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.framework import op_def_registry


tf.disable_eager_execution()


DEFAULT_EXPORT_BASE_DIR = "unit_tests"
DEFAULT_REPORT_NAME = "generation_report.json"
LIST_LENGTH = 2


def _is_excluded_op(op_name):
  lower_name = op_name.lower()
  return "sparse" in lower_name and "dense" in lower_name


def _to_snake_case(name):
  name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
  return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _model_dir(export_base_dir, op_name):
  return os.path.join(export_base_dir, "model_{}".format(op_name.upper()))


def _placeholder_name(name):
  return re.sub("[^A-Za-z0-9_]", "_", name)


def _safe_np_dtype(dtype):
  if dtype == tf.string:
    return np.object_
  if dtype == tf.bool:
    return np.bool_
  return dtype.as_numpy_dtype


def _dynamic_shape_for_input(input_name, dtype):
  lower_name = input_name.lower()
  if any(token in lower_name for token in ("axis", "dim", "depth",
                                           "num_split", "num_segments")):
    return []
  if any(token in lower_name for token in ("shape", "begin", "end", "size",
                                           "stride", "perm", "multiples")):
    return [None]
  if dtype == tf.resource:
    return []
  if dtype == tf.string:
    return [None]
  return [None, 4]


def _feed_value_for_input(input_name, dtype, shape):
  lower_name = input_name.lower()
  if shape == []:
    if dtype == tf.bool:
      return np.array(True, dtype=_safe_np_dtype(dtype))
    if dtype == tf.string:
      return np.array(b"value", dtype=np.object_)
    return np.array(1, dtype=_safe_np_dtype(dtype))

  if any(token in lower_name for token in ("shape", "size")):
    return np.array([1, 4], dtype=_safe_np_dtype(dtype))
  if "begin" in lower_name:
    return np.array([0, 0], dtype=_safe_np_dtype(dtype))
  if "end" in lower_name:
    return np.array([1, 4], dtype=_safe_np_dtype(dtype))
  if "stride" in lower_name:
    return np.array([1, 1], dtype=_safe_np_dtype(dtype))
  if "perm" in lower_name:
    return np.array([1, 0], dtype=_safe_np_dtype(dtype))
  if "multiple" in lower_name:
    return np.array([1, 1], dtype=_safe_np_dtype(dtype))
  if "indices" in lower_name or "ids" in lower_name:
    return np.array([0, 0], dtype=_safe_np_dtype(dtype))

  if dtype == tf.bool:
    return np.array([[True, False, True, False]], dtype=np.bool_)
  if dtype == tf.string:
    return np.array([b"alpha", b"beta"], dtype=np.object_)
  if dtype.is_integer:
    return np.array([[1, 2, 3, 4]], dtype=_safe_np_dtype(dtype))
  if dtype.is_complex:
    return np.array([[1 + 1j, 2 + 0j, 3 - 1j, 4 + 2j]],
                    dtype=_safe_np_dtype(dtype))
  return np.array([[1.0, 2.0, 3.0, 4.0]], dtype=_safe_np_dtype(dtype))


def _is_supported_dtype(dtype):
  return dtype not in (tf.resource, tf.variant)


def _allowed_type_values(attr_def):
  values = attr_def.allowed_values.list.type
  if values:
    return [tf.dtypes.as_dtype(value) for value in values]
  return []


def _choose_dtype(attr_def=None):
  preferred = [
      tf.float32, tf.int32, tf.int64, tf.bool, tf.string, tf.complex64,
      tf.float64
  ]
  allowed = _allowed_type_values(attr_def) if attr_def is not None else []
  if allowed:
    for dtype in preferred:
      if dtype in allowed and _is_supported_dtype(dtype):
        return dtype
    raise ValueError("no supported dtype in allowed type list")
  return tf.float32


def _attr_default(attr_def):
  if not attr_def.HasField("default_value"):
    return None

  value = attr_def.default_value
  if attr_def.type == "string":
    return value.s
  if attr_def.type == "int":
    return value.i
  if attr_def.type == "float":
    return value.f
  if attr_def.type == "bool":
    return value.b
  if attr_def.type == "type":
    return tf.dtypes.as_dtype(value.type)
  if attr_def.type == "shape":
    return _tensor_shape_from_proto(value.shape)
  if attr_def.type == "list(string)":
    return list(value.list.s)
  if attr_def.type == "list(int)":
    return list(value.list.i)
  if attr_def.type == "list(float)":
    return list(value.list.f)
  if attr_def.type == "list(bool)":
    return list(value.list.b)
  if attr_def.type == "list(type)":
    return [tf.dtypes.as_dtype(dtype) for dtype in value.list.type]
  if attr_def.type == "list(shape)":
    return [_tensor_shape_from_proto(shape) for shape in value.list.shape]
  return None


def _tensor_shape_from_proto(shape):
  if shape.unknown_rank:
    return tf.TensorShape(None)
  return tf.TensorShape([
      None if dim.size == -1 else dim.size
      for dim in shape.dim
  ])


def _synthesized_attr(attr_def):
  default_value = _attr_default(attr_def)
  if default_value is not None:
    return default_value

  if attr_def.type == "string":
    return b""
  if attr_def.type == "int":
    minimum = attr_def.minimum
    return max(minimum, 1)
  if attr_def.type == "float":
    return 1.0
  if attr_def.type == "bool":
    return False
  if attr_def.type == "type":
    return _choose_dtype(attr_def)
  if attr_def.type == "shape":
    return tf.TensorShape([None, 4])
  if attr_def.type == "list(string)":
    return [b"", b""]
  if attr_def.type == "list(int)":
    minimum = attr_def.minimum
    return [max(minimum, 1)] * LIST_LENGTH
  if attr_def.type == "list(float)":
    return [1.0] * LIST_LENGTH
  if attr_def.type == "list(bool)":
    return [False] * LIST_LENGTH
  if attr_def.type == "list(type)":
    return [_choose_dtype(attr_def)] * LIST_LENGTH
  if attr_def.type == "list(shape)":
    return [tf.TensorShape([None, 4])] * LIST_LENGTH
  if attr_def.type.startswith("func"):
    raise ValueError("function attrs are not synthesized")

  raise ValueError("unsupported attr type {}".format(attr_def.type))


def _attr_defs_by_name(op_def):
  return {attr.name: attr for attr in op_def.attr}


def _required_attrs_used_by_inputs(op_def):
  used = set()
  for arg in op_def.input_arg:
    if arg.type_attr:
      used.add(arg.type_attr)
    if arg.number_attr:
      used.add(arg.number_attr)
    if arg.type_list_attr:
      used.add(arg.type_list_attr)
  return used


def _placeholder_for_arg(arg, dtype, suffix=""):
  placeholder_name = _placeholder_name(arg.name + suffix)
  shape = _dynamic_shape_for_input(arg.name, dtype)
  tensor = tf.placeholder(dtype=dtype, shape=shape, name=placeholder_name)
  feed = _feed_value_for_input(arg.name, dtype, shape)
  return tensor, feed


def _dtype_for_arg(arg, attr_defs, attr_values):
  if arg.type:
    dtype = tf.dtypes.as_dtype(arg.type)
  elif arg.type_attr:
    if arg.type_attr not in attr_values:
      attr_values[arg.type_attr] = _choose_dtype(attr_defs[arg.type_attr])
    dtype = attr_values[arg.type_attr]
  else:
    dtype = tf.float32

  if not _is_supported_dtype(dtype):
    raise ValueError("unsupported input dtype {}".format(dtype.name))
  return dtype


def _build_generic_raw_op(raw_op_name):
  raw_op = getattr(tf.raw_ops, raw_op_name)
  op_def = op_def_registry.get(raw_op_name)
  if op_def is None:
    raise ValueError("missing OpDef")

  attr_defs = _attr_defs_by_name(op_def)
  attr_values = {}
  kwargs = {}
  inputs = {}
  feeds = {}

  for attr_name in _required_attrs_used_by_inputs(op_def):
    attr_def = attr_defs[attr_name]
    if attr_def.type == "type":
      attr_values[attr_name] = _choose_dtype(attr_def)
    elif attr_def.type == "list(type)":
      attr_values[attr_name] = [_choose_dtype(attr_def)] * LIST_LENGTH
    elif attr_def.type == "int":
      attr_values[attr_name] = max(attr_def.minimum, LIST_LENGTH)

  for arg in op_def.input_arg:
    if arg.number_attr:
      count = attr_values.get(arg.number_attr, LIST_LENGTH)
      dtype = _dtype_for_arg(arg, attr_defs, attr_values)
      tensors = []
      for i in range(count):
        tensor, feed = _placeholder_for_arg(arg, dtype, "_{}".format(i))
        tensors.append(tensor)
        inputs["{}_{}".format(arg.name, i)] = tensor
        feeds[tensor] = feed
      kwargs[arg.name] = tensors
    elif arg.type_list_attr:
      dtypes = attr_values.get(arg.type_list_attr,
                               [tf.float32] * LIST_LENGTH)
      tensors = []
      for i, dtype in enumerate(dtypes):
        tensor, feed = _placeholder_for_arg(arg, dtype, "_{}".format(i))
        tensors.append(tensor)
        inputs["{}_{}".format(arg.name, i)] = tensor
        feeds[tensor] = feed
      kwargs[arg.name] = tensors
    else:
      dtype = _dtype_for_arg(arg, attr_defs, attr_values)
      tensor, feed = _placeholder_for_arg(arg, dtype)
      inputs[arg.name] = tensor
      feeds[tensor] = feed
      kwargs[arg.name] = tensor

  for attr in op_def.attr:
    if attr.name in attr_values:
      continue
    if attr.name in kwargs:
      continue
    attr_values[attr.name] = _synthesized_attr(attr)

  signature = inspect.signature(raw_op)
  for name in signature.parameters:
    if name in kwargs or name == "name":
      continue
    if name in attr_values:
      kwargs[name] = attr_values[name]

  outputs = raw_op(name=_to_snake_case(raw_op_name), **kwargs)
  if isinstance(outputs, (list, tuple)):
    if not outputs:
      raise ValueError("op produced no outputs")
    output = outputs[0]
  else:
    output = outputs

  if not hasattr(output, "dtype"):
    raise ValueError("first output is not a tensor")
  if output.dtype == tf.resource:
    raise ValueError("resource outputs are not exported as tensor results")

  return inputs, output, feeds


def _generic_raw_op_generator(raw_op_name):
  def _build():
    return _build_generic_raw_op(raw_op_name)

  _build.__name__ = "generic_raw_op"
  return _build


def _manual_builder(raw_op_name):
  builders = {
      "DynamicStitch": _build_dynamic_stitch,
      "Transpose": _build_transpose,
  }
  return builders.get(raw_op_name)


def _build_transpose():
  x = tf.placeholder(dtype=tf.float32, shape=[None, 4], name="x")
  perm = tf.placeholder(dtype=tf.int32, shape=[2], name="perm")
  output = tf.raw_ops.Transpose(x=x, perm=perm, name="transpose")
  inputs = {
      "x": x,
      "perm": perm,
  }
  feeds = {
      x: np.array([[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
      perm: np.array([1, 0], dtype=np.int32),
  }
  return inputs, output, feeds


def _build_dynamic_stitch():
  indices_0 = tf.placeholder(dtype=tf.int32, shape=[None], name="indices_0")
  indices_1 = tf.placeholder(dtype=tf.int32, shape=[None], name="indices_1")
  data_0 = tf.placeholder(dtype=tf.float32, shape=[None, 4], name="data_0")
  data_1 = tf.placeholder(dtype=tf.float32, shape=[None, 4], name="data_1")
  output = tf.raw_ops.DynamicStitch(
      indices=[indices_0, indices_1],
      data=[data_0, data_1],
      name="dynamic_stitch")
  inputs = {
      "indices_0": indices_0,
      "indices_1": indices_1,
      "data_0": data_0,
      "data_1": data_1,
  }
  feeds = {
      indices_0: np.array([0, 2], dtype=np.int32),
      indices_1: np.array([1, 3], dtype=np.int32),
      data_0: np.array([[1.0, 2.0, 3.0, 4.0],
                        [9.0, 10.0, 11.0, 12.0]], dtype=np.float32),
      data_1: np.array([[5.0, 6.0, 7.0, 8.0],
                        [13.0, 14.0, 15.0, 16.0]], dtype=np.float32),
  }
  return inputs, output, feeds


def _raw_op_names():
  names = []
  for name in dir(tf.raw_ops):
    if name.startswith("_"):
      continue
    if _is_excluded_op(name):
      continue
    raw_op = getattr(tf.raw_ops, name)
    if callable(raw_op) and op_def_registry.get(name) is not None:
      names.append(name)
  return sorted(set(names))


def _build_ops_table():
  return {
      i: (name, _manual_builder(name) or _generic_raw_op_generator(name))
      for i, name in enumerate(_raw_op_names())
  }


OPS = _build_ops_table()


def _print_ops_table():
  width = len(str(max(OPS))) if OPS else 1
  print("{:>{width}}  {:<40}  {}".format(
      "ID", "OP_NAME", "GENERATOR", width=width))
  print("{}  {}  {}".format("-" * width, "-" * 40, "-" * 32))
  for op_id, (op_name, op_builder) in sorted(OPS.items()):
    generator_name = getattr(op_builder, "__name__", "generic_raw_op")
    print("{:>{width}}  {:<40}  {}".format(
        op_id, op_name, generator_name, width=width))


def _build_signature(inputs, result):
  tensor_inputs = {
      key: tf.saved_model.utils.build_tensor_info(tensor)
      for key, tensor in sorted(inputs.items())
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


def _run_op(op_name, op_builder, args):
  export_dir = os.path.join(_model_dir(args.export_base_dir, op_name), "1")

  with tf.Graph().as_default():
    inputs, result, feeds = op_builder()
    signature = _build_signature(inputs, result)

    with tf.Session() as sess:
      if args.save_model:
        _save_model(sess, export_dir, signature)
        return {"op": op_name, "status": "saved", "path": export_dir}

      output = sess.run(result, feed_dict=feeds)
      return {
          "op": op_name,
          "status": "ran",
          "shape": list(np.shape(output)),
          "dtype": str(getattr(output, "dtype", type(output))),
      }


def _write_report(args, results):
  report_path = os.path.join(args.export_base_dir, args.report_name)
  if not os.path.isdir(args.export_base_dir):
    os.makedirs(args.export_base_dir)
  with open(report_path, "w") as report_file:
    json.dump(results, report_file, indent=2, sort_keys=True)
  return report_path


def _parse_args():
  parser = argparse.ArgumentParser(
      description=("Generate dynamic-input, one-operation TensorFlow v1 "
                   "SavedModels from the TensorFlow raw op registry."))
  parser.add_argument(
      "--op_id",
      type=int,
      default=0,
      choices=sorted(OPS.keys()),
      help="Raw-op id to generate. Use --list_ops to see available ids.")
  parser.add_argument(
      "--op_name",
      help="Raw TensorFlow op name to generate, e.g. Add or MatMul.")
  parser.add_argument(
      "--all_ops",
      action="store_true",
      help="Attempt to generate every registered TensorFlow raw op.")
  parser.add_argument(
      "--save_model",
      action="store_true",
      help="Save the selected one-operation model instead of executing it.")
  parser.add_argument(
      "--export_base_dir",
      default=DEFAULT_EXPORT_BASE_DIR,
      help="Base directory for SavedModel exports.")
  parser.add_argument(
      "--report_name",
      default=DEFAULT_REPORT_NAME,
      help="JSON report filename written under --export_base_dir.")
  parser.add_argument(
      "--list_ops",
      action="store_true",
      help="List available raw-op ids and exit.")
  return parser.parse_args()


def main():
  args = _parse_args()

  if args.list_ops:
    _print_ops_table()
    return

  if args.all_ops:
    op_specs = [OPS[i] for i in sorted(OPS)]
  elif args.op_name:
    if _is_excluded_op(args.op_name):
      raise ValueError("{} is excluded from generation".format(args.op_name))
    op_specs = [(args.op_name,
                 _manual_builder(args.op_name) or
                 _generic_raw_op_generator(args.op_name))]
  else:
    op_specs = [OPS[args.op_id]]

  results = {"generated": [], "skipped": []}
  for op_name, op_builder in op_specs:
    try:
      result = _run_op(op_name, op_builder, args)
      results["generated"].append(result)
      print("{} {}".format(result["status"], op_name))
    except Exception as exc:  # pylint: disable=broad-except
      skip = {"op": op_name, "reason": str(exc)}
      results["skipped"].append(skip)
      print("skipped {}: {}".format(op_name, exc))

  report_path = _write_report(args, results)
  print("Generated: {}".format(len(results["generated"])))
  print("Skipped: {}".format(len(results["skipped"])))
  print("Report: {}".format(report_path))
  print("Done")


if __name__ == "__main__":
  main()
