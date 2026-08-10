#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_SHAPE_EXPR_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_SHAPE_EXPR_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "xla/shape_expr.h"

namespace tensorflow {

// TensorFlow shape inference and XLA use the same owning expression value.
// TensorFlow-specific helpers below only bridge TensorShapeProto's protobuf.
using DimExpr = xla::DExpr;

DimExpr DimExprFromProto(const ExpressionProto& proto);
void DimExprToProto(const DimExpr& expr, ExpressionProto* proto);
std::string DimExprDebugString(const DimExpr& expr);

// Simplifies through xla::DExpr and stores the returned value in `arena`.
DimExpr* SimplifyExpr(DimExpr* expr,
                      std::vector<std::unique_ptr<DimExpr>>* arena);

// Returns whether TensorShape should preserve symbolic expressions. The
// shape-expression support follows the `tf_xla_enable_dynamic_sizes` flag.
bool TensorShapeExpressionsEnabled();

// Overrides TensorShapeExpressionsEnabled for tests. Passing std::nullopt
// restores the default environment-derived behavior.
void SetTensorShapeExpressionsEnabledForTesting(std::optional<bool> enabled);

// Returns true if the expression proto depends on a symbolic variable.
bool IsDynamicDimExpr(const ExpressionProto& proto);

// Returns true if any expression attached to the TensorShapeProto is dynamic.
bool HasDynamicDimExprs(const TensorShapeProto& proto);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_SHAPE_EXPR_H_
