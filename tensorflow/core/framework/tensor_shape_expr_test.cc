/* Copyright 2026 The TensorFlow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/tensor_shape_expr.h"

#include <memory>
#include <optional>
#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class TensorShapeExpressionsEnabledTest : public ::testing::Test {
 protected:
  void SetUp() override {
    SetTensorShapeExpressionsEnabledForTesting(true);
  }

  void TearDown() override {
    SetTensorShapeExpressionsEnabledForTesting(std::nullopt);
  }
};

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

TEST_F(TensorShapeExpressionsEnabledTest,
       SetExpressionsRejectsEntriesBeyondRank) {
  TensorShape shape({2, 3});
  EXPECT_DEATH(
      shape.set_expressions(
          {xla::DExpr::Var(1), xla::DExpr::Var(2), xla::DExpr::Var(3)}),
      "");
}

TEST_F(TensorShapeExpressionsEnabledTest,
       RemoveDimRangePreservesRemainingExpressions) {
  TensorShape shape({2, 3});
  shape.set_expressions({xla::DExpr::Var(1), xla::DExpr::Var(2)});

  shape.RemoveDim(0);

  ASSERT_EQ(shape.get_expressions().size(), 1);
  EXPECT_TRUE(shape.get_expression(0) == xla::DExpr::Var(2));
}

}  // namespace
}  // namespace tensorflow
