#include "tensorflow/core/framework/tensor_shape_expr.h"

#include <memory>
#include <vector>

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(TensorShapeExprTest, UsesXlaCanonicalization) {
  ExpressionProto proto;
  auto* add = proto.mutable_add_node();
  add->mutable_lhs()->mutable_div_node()->mutable_lhs()->set_variable_id(1);
  add->mutable_lhs()->mutable_div_node()->mutable_rhs()->set_constant_value(2);
  add->mutable_rhs()->mutable_div_node()->mutable_lhs()->set_variable_id(1);
  add->mutable_rhs()->mutable_div_node()->mutable_rhs()->set_constant_value(2);

  auto expr = DimExprFromProto(proto);
  EXPECT_EQ(DimExprDebugString(expr.simplify()), "A");
}

TEST(TensorShapeExprTest, TensorFlowProtoRoundTripsSharedExpression) {
  DimExpr original = (DimExpr::Var(7) + DimExpr::Const(3)) / 2;
  ExpressionProto proto;
  DimExprToProto(original, &proto);

  auto round_tripped = DimExprFromProto(proto);
  EXPECT_TRUE(xla::DynExpr::equal(original.get(), round_tripped.get()));
}

TEST(TensorShapeExprTest, TensorFlowProtoRoundTripsConditionalExpression) {
  DimExpr variable = DimExpr::Var(7);
  DimExpr original = DimExpr::Select(
      DimExpr::Gt(variable, DimExpr::Const(3)),
      DimExpr::Max(variable, DimExpr::Const(8)),
      (variable + DimExpr::Const(3)) / 2);
  ExpressionProto proto;
  DimExprToProto(original, &proto);

  auto round_tripped = DimExprFromProto(proto);
  EXPECT_TRUE(xla::DynExpr::equal(original.get(), round_tripped.get()));
}

TEST(TensorShapeExprTest, SimplifyExprUsesSharedImplementation) {
  DimExpr original =
      (DimExpr::Var(1) / 2) + (DimExpr::Var(1) / 2);
  std::vector<std::unique_ptr<DimExpr>> arena;

  DimExpr* simplified = SimplifyExpr(&original, &arena);

  ASSERT_NE(simplified, nullptr);
  EXPECT_EQ(DimExprDebugString(*simplified), "A");
}

}  // namespace
}  // namespace tensorflow
