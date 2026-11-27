/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/dynamic_shape_bounds.h"

#include <string>
#include <vector>

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using xla::DExpr;

absl::Status CheckDimension(const DExpr& expr) {
  const std::vector<DExpr> dimensions = {expr};
  return CheckDynamicShapeBounds(dimensions, "input");
}

TEST(DynamicShapeBoundsTest, AcceptsIncreasingDimensions) {
  const DExpr a = DExpr::Var(1);
  const std::vector<DExpr> expressions = {
      a, a + 4, a - 4, 3 * a, a / 240, (3 * a + 2) / 3,
      a * a, DExpr::Max(a - 10, DExpr::Const(0)),
      DExpr::Gt(a, DExpr::Const(10)),
      DExpr::Select(DExpr::Const(1), a, DExpr::Const(100) - a)};
  for (const auto& expr : expressions) {
    EXPECT_TRUE(CheckDimension(expr).ok());
  }
}

TEST(DynamicShapeBoundsTest, RejectsDecreasingAndMixedDimensions) {
  const DExpr a = DExpr::Var(1);
  const DExpr remaining = DExpr::Const(100) - a;
  const std::vector<DExpr> expressions = {
      remaining, DExpr::Const(100) / a, DExpr::Const(100) + (-1 * a), a / -2,
      (remaining + 4) / 2, DExpr::Max(remaining, DExpr::Const(0)),
      a * remaining, (a - 100) * (a - 100),
      DExpr::Gt(DExpr::Const(100), a),
      DExpr::Select(DExpr::Gt(a, DExpr::Const(10)), DExpr::Const(1), a)};
  for (const auto& expr : expressions) {
    const auto status = CheckDimension(expr);
    EXPECT_TRUE(IsDynamicShapeBoundsError(status)) << status;
    EXPECT_NE(status.message().find("input dimension 0"), std::string::npos);
  }
}

TEST(DynamicShapeBoundsTest, CoversSmallerRequestsReusingAnExistingBucket) {
  const DExpr a = DExpr::Var(1);
  const DExpr remaining = DExpr::Const(100) - a;
  // Even if the first request exactly matches its bound of 20, reuse for A=5
  // would need 95 elements in a buffer allocated for 80. Checking only the
  // current request against the padded value would miss this later request.
  EXPECT_EQ(remaining.substitute(1, DExpr::Const(20))->get_val(), 80);
  EXPECT_EQ(remaining.substitute(1, DExpr::Const(5))->get_val(), 95);
  EXPECT_TRUE(IsDynamicShapeBoundsError(CheckDimension(remaining)));
}

TEST(DynamicShapeBoundsTest, KeepsStaticAndMissingExpressionMetadata) {
  const std::vector<DExpr> dimensions = {
      DExpr(), DExpr::Unknown(), DExpr::Const(80), DExpr::Const(0)};
  EXPECT_TRUE(CheckDynamicShapeBounds(dimensions, "static input").ok());
  EXPECT_FALSE(IsDynamicShapeBoundsError(absl::InvalidArgumentError("other")));
}

TEST(DynamicShapeBoundsTest, RejectsUnknownOrInvalidTermsInsideDynamicShapes) {
  const DExpr a = DExpr::Var(1);
  EXPECT_TRUE(IsDynamicShapeBoundsError(CheckDimension(a + DExpr::Unknown())));
  EXPECT_TRUE(IsDynamicShapeBoundsError(CheckDimension(a / 0)));
}

void SetVectorShape(xla::ShapeProto* shape, const DExpr& expr) {
  shape->set_element_type(xla::F32);
  shape->add_dimensions(80);
  expr.to_proto(shape->add_expressions());
}

TEST(DynamicShapeBoundsTest, ChecksIntermediateBuffersWithStaticOutputs) {
  xla::HloModuleProto module;
  auto* computation = module.add_computations();
  computation->set_name("entry");
  auto* input = computation->add_instructions();
  input->set_name("input");
  SetVectorShape(input->mutable_shape(), DExpr::Var(1));
  auto* internal = computation->add_instructions();
  internal->set_name("fill");
  SetVectorShape(internal->mutable_shape(), DExpr::Const(100) - DExpr::Var(1));
  auto* output = computation->add_instructions();
  output->set_name("sum");
  output->mutable_shape()->set_element_type(xla::F32);

  // The unsafe allocation is hidden by the scalar result of a reduction.
  const auto status = CheckDynamicShapeBounds(module);
  EXPECT_TRUE(IsDynamicShapeBoundsError(status));
  EXPECT_NE(status.message().find("entry/fill"), std::string::npos);
}

TEST(DynamicShapeBoundsTest, ChecksNestedComputationsAndTupleLeaves) {
  xla::HloModuleProto module;
  module.add_computations()->set_name("entry");
  auto* nested = module.add_computations();
  nested->set_name("body");
  auto* output = nested->add_instructions();
  output->set_name("tuple");
  auto* shape = output->mutable_shape();
  shape->set_element_type(xla::TUPLE);
  SetVectorShape(shape->add_tuple_shapes(), DExpr::Const(100) / DExpr::Var(1));

  const auto status = CheckDynamicShapeBounds(module);
  EXPECT_TRUE(IsDynamicShapeBoundsError(status));
  EXPECT_NE(status.message().find("body/tuple tuple element 0"),
            std::string::npos);
}

TEST(DynamicShapeBoundsTest, AllowsDecreasingValuesInScalarContents) {
  xla::HloModuleProto module;
  auto* instruction = module.add_computations()->add_instructions();
  instruction->mutable_shape()->set_element_type(xla::S32);
  // Only use as a dimension can make this scalar shrink an allocation.
  (DExpr::Const(100) - DExpr::Var(1))
      .to_proto(instruction->add_contents());
  EXPECT_TRUE(CheckDynamicShapeBounds(module).ok());
}

}  // namespace
}  // namespace tensorflow
