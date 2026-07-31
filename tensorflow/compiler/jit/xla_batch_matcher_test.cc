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

#include "tensorflow/compiler/jit/xla_batch_matcher.h"

#include <cstdint>
#include <vector>

#include "tensorflow/core/platform/test.h"
#include "xla/shape_expr.h"

namespace tensorflow {
namespace {

TEST(XlaBatchMatcherTest, AlignsBucketForDynamicReshapeDivision) {
  std::vector<xla::DExpr> reshape_expressions = {
      xla::DExpr::Var(1) / 12};

  const int64_t alignment =
      GetDynamicPaddingAlignment(reshape_expressions);

  EXPECT_EQ(GetAlignedPowerOfTwoBatch(300, /*alignment=*/1), 512);
  EXPECT_EQ(alignment, 12);
  EXPECT_EQ(GetAlignedPowerOfTwoBatch(300, alignment), 384);
  EXPECT_EQ(GetAlignedPowerOfTwoBatch(384, alignment), 384);
  EXPECT_EQ(GetAlignedPowerOfTwoBatch(385, alignment), 768);
}

TEST(XlaBatchMatcherTest, CombinesIndependentReshapeDivisors) {
  std::vector<xla::DExpr> reshape_expressions = {
      xla::DExpr::Var(1) / 12, xla::DExpr::Var(1) / 18};

  const int64_t alignment =
      GetDynamicPaddingAlignment(reshape_expressions);

  EXPECT_EQ(alignment, 36);
  EXPECT_EQ(GetAlignedPowerOfTwoBatch(300, alignment), 576);
}

}  // namespace
}  // namespace tensorflow
