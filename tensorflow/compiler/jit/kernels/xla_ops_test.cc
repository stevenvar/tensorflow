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

#include "tensorflow/compiler/jit/kernels/xla_ops_internal.h"

#include <cstdint>
#include <string>
#include <vector>

#include "xla/shape_expr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace xla_ops_internal {
namespace {

XlaArgument MakeDynamicArgument(int64_t observed_size) {
  XlaArgument arg;
  arg.kind = XlaArgument::kParameter;
  arg.type = DT_FLOAT;
  TensorShape shape({observed_size, 16});
  shape.set_expression(0, xla::DExpr::Var(1));
  shape.set_expression(1, xla::DExpr::Const(16));
  arg.shape = shape;
  return arg;
}

TEST(DynamicSolveFilterTest, IgnoresSingletonWhenNonSingletonEvidenceExists) {
  std::vector<XlaArgument> args = {
      MakeDynamicArgument(1),
      MakeDynamicArgument(240),
  };

  DynamicSolveFilterDecision decision =
      AnalyzeIgnoredDynamicArgumentOccurrences(args);

  ASSERT_TRUE(decision.can_run);
  ASSERT_EQ(decision.ignored_occurrences.size(), 1);
  EXPECT_EQ(decision.ignored_occurrences[0].arg_index, 0);
  EXPECT_EQ(decision.ignored_occurrences[0].observed_value, 1);

  StripIgnoredDynamicArgumentOccurrences(decision.ignored_occurrences, &args);
  const TensorShape& singleton_shape = std::get<TensorShape>(args[0].shape);
  const TensorShape& nonsingleton_shape = std::get<TensorShape>(args[1].shape);
  EXPECT_FALSE(singleton_shape.get_expression(0));
  EXPECT_EQ(singleton_shape.dim_size(0), 1);
  EXPECT_TRUE(nonsingleton_shape.get_expression(0));
}

TEST(DynamicSolveFilterTest, RejectsConflictingNonSingletonEvidence) {
  std::vector<XlaArgument> args = {
      MakeDynamicArgument(240),
      MakeDynamicArgument(720),
  };

  DynamicSolveFilterDecision decision =
      AnalyzeIgnoredDynamicArgumentOccurrences(args);

  EXPECT_FALSE(decision.can_run);
  EXPECT_TRUE(decision.ignored_occurrences.empty());
  EXPECT_NE(decision.diagnostic.find("conflicting non-singleton candidates"),
            std::string::npos);
}

TEST(DynamicSolveFilterTest, KeepsConsistentSingletonOnlyEvidence) {
  std::vector<XlaArgument> args = {
      MakeDynamicArgument(1),
      MakeDynamicArgument(1),
  };

  DynamicSolveFilterDecision decision =
      AnalyzeIgnoredDynamicArgumentOccurrences(args);

  EXPECT_TRUE(decision.can_run);
  EXPECT_TRUE(decision.ignored_occurrences.empty());
}

}  // namespace
}  // namespace xla_ops_internal
}  // namespace tensorflow
