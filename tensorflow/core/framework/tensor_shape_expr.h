#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_SHAPE_EXPR_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_SHAPE_EXPR_H_

#include <string>

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "xla/shape_dynexpr.h"

namespace tensorflow {

using DExpr = xla::DExpr;

// Converts TensorFlow shape-expression protos to the owning DExpr wrapper used
// across XLA and TensorFlow shape propagation.
DExpr DExprFromProto(const ExpressionProto& proto);

// Serializes a TensorFlow DExpr back into TensorShapeProto expression form.
void ExprToProto(const DExpr& expr, ExpressionProto* proto);

// Renders an expression for logging/debug output.
std::string ExprToString(const DExpr& expr);

// Structural equality check that also distinguishes empty and unknown values.
bool ExprEquals(const DExpr& a, const DExpr& b);

inline DExpr SimplifyExpr(const DExpr& expr) { return expr.simplify(); }

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_SHAPE_EXPR_H_
