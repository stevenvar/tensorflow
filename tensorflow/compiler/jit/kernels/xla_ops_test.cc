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
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "xla/shape_expr.h"
#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace xla_ops_internal {
namespace {

TEST(XlaInputMappingTest, UsesCompilationInputMapping) {
  const std::vector<int> input_mapping = {1, 3};

  EXPECT_EQ(GetRuntimeInputIndex(input_mapping, 0, /*num_constant_args=*/1,
                                 /*constants_omitted=*/false),
            1);
  EXPECT_EQ(GetRuntimeInputIndex(input_mapping, 1, /*num_constant_args=*/1,
                                 /*constants_omitted=*/false),
            3);
  EXPECT_EQ(GetRuntimeInputIndex(input_mapping, 0, /*num_constant_args=*/1,
                                 /*constants_omitted=*/true),
            0);
  EXPECT_EQ(GetRuntimeInputIndex(input_mapping, 1, /*num_constant_args=*/1,
                                 /*constants_omitted=*/true),
            2);
  EXPECT_EQ(GetRuntimeInputIndex(input_mapping, 2, /*num_constant_args=*/1,
                                 /*constants_omitted=*/true),
            -1);
}

XlaArgument MakeDynamicArgument(int64_t observed_size,
                                xla::DExpr expr = xla::DExpr::Var(1)) {
  XlaArgument arg;
  arg.kind = XlaArgument::kParameter;
  arg.type = DT_FLOAT;
  TensorShape shape({observed_size, 16});
  shape.set_expression(0, std::move(expr));
  shape.set_expression(1, xla::DExpr::Const(16));
  arg.shape = shape;
  return arg;
}

XlaArgument MakeDynamicConstantArgument(int32 observed_value) {
  XlaArgument arg;
  arg.kind = XlaArgument::kConstant;
  arg.type = DT_INT32;
  arg.shape = TensorShape({1});
  arg.constant_value = Tensor(DT_INT32, TensorShape({1}));
  arg.constant_value.flat<int32>()(0) = observed_value;
  arg.constant_value_expressions.resize(1);
  arg.constant_value_expressions[0].set_variable_id(1);
  return arg;
}

XlaArgument MakeConstantContents(DataType dtype,
                                const std::vector<int64_t>& values,
                                const std::vector<xla::DExpr>& expressions) {
  XlaArgument arg;
  arg.kind = XlaArgument::kConstant;
  arg.type = dtype;
  arg.shape = TensorShape({static_cast<int64_t>(values.size())});
  arg.constant_value = Tensor(dtype, std::get<TensorShape>(arg.shape));
  for (int i = 0; i < values.size(); ++i) {
    if (dtype == DT_INT32) {
      arg.constant_value.flat<int32>()(i) = values[i];
    } else {
      arg.constant_value.flat<int64_t>()(i) = values[i];
    }
  }
  for (const xla::DExpr& expr : expressions) {
    xla::ExpressionProto proto;
    expr.to_proto(&proto);
    arg.constant_value_expressions.push_back(std::move(proto));
  }
  return arg;
}

std::vector<int64_t> ConstantValues(const XlaArgument& arg) {
  std::vector<int64_t> values;
  for (int64_t i = 0; i < arg.constant_value.NumElements(); ++i) {
    values.push_back(arg.type == DT_INT32
                         ? arg.constant_value.flat<int32>()(i)
                         : arg.constant_value.flat<int64_t>()(i));
  }
  return values;
}

class ConstantNormalizationTest : public ::testing::TestWithParam<DataType> {};

TEST_P(ConstantNormalizationTest, PreservesEqualLiteralsAndRuntimeTensor) {
  const XlaArgument original = MakeConstantContents(
      GetParam(), {10, 10, 10},
      {xla::DExpr::Var(1), xla::DExpr::Const(10), xla::DExpr::Unknown()});
  XlaArgument normalized = original;

  // Equal numeric values have different meanings. Only the marked A changes.
  auto status = NormalizeConstantArgument(20, &normalized);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(ConstantValues(normalized), (std::vector<int64_t>{20, 10, 10}));
  EXPECT_EQ(ConstantValues(original), (std::vector<int64_t>{10, 10, 10}));
  for (int i = 0; i < original.constant_value_expressions.size(); ++i) {
    EXPECT_EQ(normalized.constant_value_expressions[i].SerializeAsString(),
              original.constant_value_expressions[i].SerializeAsString());
  }
}

TEST_P(ConstantNormalizationTest, LeavesUnmarkedConstantsAlone) {
  XlaArgument arg = MakeConstantContents(GetParam(), {10}, {});
  const auto original_data = arg.constant_value.tensor_data().data();
  auto status = NormalizeConstantArgument(20, &arg);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(ConstantValues(arg), (std::vector<int64_t>{10}));
  EXPECT_TRUE(arg.constant_value_expressions.empty());
  EXPECT_EQ(arg.constant_value.tensor_data().data(), original_data);
}

TEST_P(ConstantNormalizationTest, LiteralContentsDoNotRequireRootBound) {
  XlaArgument arg = MakeConstantContents(GetParam(), {10}, {xla::DExpr::Const(10)});
  auto status = NormalizeConstantArgument(0, &arg);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(ConstantValues(arg), (std::vector<int64_t>{10}));
}

TEST_P(ConstantNormalizationTest, EvaluatesDerivedAndRepeatedContents) {
  const xla::DExpr a = xla::DExpr::Var(1);
  XlaArgument arg = MakeConstantContents(GetParam(), {3, 4, 720, 720},
                                        {a / 240, a / 240 + 1, a, a});
  auto status = NormalizeConstantArgument(1440, &arg);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(ConstantValues(arg), (std::vector<int64_t>{6, 7, 1440, 1440}));
}

TEST_P(ConstantNormalizationTest, AllowsNegativeIntegerContents) {
  // Tensor contents may be negative even though allocation dimensions cannot.
  XlaArgument arg = MakeConstantContents(
      GetParam(), {-9}, {xla::DExpr::Const(1) - xla::DExpr::Var(1)});
  auto status = NormalizeConstantArgument(20, &arg);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(ConstantValues(arg), (std::vector<int64_t>{-19}));
}

TEST_P(ConstantNormalizationTest, DoesNotCopyUnchangedTensor) {
  XlaArgument arg = MakeConstantContents(GetParam(), {20}, {xla::DExpr::Var(1)});
  const auto original_data = arg.constant_value.tensor_data().data();
  auto status = NormalizeConstantArgument(20, &arg);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(arg.constant_value.tensor_data().data(), original_data);
}

TEST_P(ConstantNormalizationTest, RejectsInvalidExpressionsWithoutPartialWrite) {
  const xla::DExpr a = xla::DExpr::Var(1);
  const std::vector<xla::DExpr> invalid = {
      xla::DExpr::Var(2), a + xla::DExpr::Var(2),
      a / xla::DExpr::Const(0), a / (a - 20), a + xla::DExpr::Unknown()};
  for (const auto& expr : invalid) {
    // The first element can be normalized, but the second must reject it all.
    XlaArgument arg = MakeConstantContents(GetParam(), {10, 10}, {a, expr});
    const auto original_data = arg.constant_value.tensor_data().data();
    auto status = NormalizeConstantArgument(20, &arg);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(ConstantValues(arg), (std::vector<int64_t>{10, 10}));
    EXPECT_EQ(arg.constant_value.tensor_data().data(), original_data);
  }
}

TEST_P(ConstantNormalizationTest, RejectsMismatchedMetadataAndMissingBound) {
  XlaArgument arg = MakeConstantContents(GetParam(), {10},
                                        {xla::DExpr::Var(1), xla::DExpr::Var(1)});
  EXPECT_FALSE(NormalizeConstantArgument(20, &arg).ok());
  arg.constant_value_expressions.resize(1);
  EXPECT_FALSE(NormalizeConstantArgument(0, &arg).ok());
  EXPECT_EQ(ConstantValues(arg), (std::vector<int64_t>{10}));
}

TEST_P(ConstantNormalizationTest, CacheKeyUsesPaddedContentsAndRetainsLiterals) {
  const xla::DExpr a = xla::DExpr::Var(1);
  std::vector<XlaArgument> first = {
      MakeConstantContents(GetParam(), {3, 720}, {a / 240, a}),
      MakeConstantContents(GetParam(), {720}, {})};
  std::vector<XlaArgument> second = {
      MakeConstantContents(GetParam(), {4, 960}, {a / 240, a}),
      MakeConstantContents(GetParam(), {720}, {})};
  for (auto* args : {&first, &second}) {
    for (auto& arg : *args) {
      auto status = NormalizeConstantArgument(1440, &arg);
      ASSERT_TRUE(status.ok()) << status;
    }
  }
  NameAttrList function;
  function.set_name("shape_consumer");
  auto first_key = DeviceCompilationClusterSignature::Build(function, first);
  auto second_key = DeviceCompilationClusterSignature::Build(function, second);
  ASSERT_TRUE(first_key.ok()) << first_key.status();
  ASSERT_TRUE(second_key.ok()) << second_key.status();
  EXPECT_EQ(*first_key, *second_key);
  EXPECT_EQ(DeviceCompilationClusterSignature::Hash()(*first_key),
            DeviceCompilationClusterSignature::Hash()(*second_key));

  // A changed ordinary constant must still require its own executable.
  second[1] = MakeConstantContents(GetParam(), {960}, {});
  auto changed_key = DeviceCompilationClusterSignature::Build(function, second);
  ASSERT_TRUE(changed_key.ok()) << changed_key.status();
  EXPECT_FALSE(*first_key == *changed_key);
}

INSTANTIATE_TEST_SUITE_P(IntegerTypes, ConstantNormalizationTest,
                        ::testing::Values(DT_INT32, DT_INT64));

TEST(ConstantNormalizationScalarTest, ChecksInt32RangeAndSupportsInt64) {
  const int64_t bound = static_cast<int64_t>(std::numeric_limits<int32>::max()) + 1;
  XlaArgument narrow = MakeDynamicConstantArgument(10);
  EXPECT_FALSE(NormalizeConstantArgument(bound, &narrow).ok());
  EXPECT_EQ(ConstantValues(narrow), (std::vector<int64_t>{10}));

  XlaArgument wide;
  wide.kind = XlaArgument::kConstant;
  wide.type = DT_INT64;
  wide.shape = TensorShape({});
  wide.constant_value = Tensor(DT_INT64, TensorShape({}));
  wide.constant_value.scalar<int64_t>()() = 10;
  wide.constant_value_expressions.resize(1);
  wide.constant_value_expressions[0].set_variable_id(1);
  auto status = NormalizeConstantArgument(bound, &wide);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(wide.constant_value.scalar<int64_t>()(), bound);
  EXPECT_EQ(wide.constant_value.dims(), 0);
}

xla::Shape MakeDynamicXlaInput(xla::DExpr leading_expr) {
  xla::Shape shape;
  shape.set_element_type(xla::F32);
  shape.add_dimensions(1024, /*is_dynamic=*/true,
                       std::move(leading_expr));
  shape.add_dimensions(16, /*is_dynamic=*/false, xla::DExpr::Const(16));
  return shape;
}

DynamicBatchResolutionResult ResolveForTest(
    absl::Span<const xla::Shape> xla_shapes,
    absl::Span<const TensorShape> runtime_shapes,
    absl::Span<const int> runtime_input_indices,
    absl::Span<const std::string> runtime_input_names) {
  return ResolveDynamicBatchSizeFromRuntimeShapes(
      xla_shapes, runtime_shapes.size(),
      [&](int i) { return runtime_input_indices[i]; },
      [&](int i) -> const TensorShape& { return runtime_shapes[i]; },
      [&](int i) -> absl::string_view { return runtime_input_names[i]; });
}

TEST(DynamicRuntimeSolveTest, UsesMappedRuntimeShapesToSolveExpressions) {
  const xla::DExpr variable = xla::DExpr::Var(1);
  const std::vector<xla::Shape> xla_shapes = {
      MakeDynamicXlaInput(variable),
      MakeDynamicXlaInput(variable / 2),
  };
  // XLA parameters are deliberately ordered differently from runtime inputs.
  const std::vector<TensorShape> runtime_shapes = {
      TensorShape({120, 16}),
      TensorShape({240, 16}),
  };
  const std::vector<int> runtime_input_indices = {1, 0};
  const std::vector<std::string> runtime_input_names = {"half", "full"};

  DynamicBatchResolutionResult result = ResolveForTest(
      xla_shapes, runtime_shapes, runtime_input_indices, runtime_input_names);

  EXPECT_TRUE(result.can_run);
  ASSERT_TRUE(result.has_batch_size);
  EXPECT_EQ(result.batch_size, 240);
}

TEST(DynamicRuntimeSolveTest,
     PrefersNonSingletonEvidenceAcrossEquivalentExpressions) {
  const xla::DExpr variable = xla::DExpr::Var(1);
  const std::vector<xla::Shape> xla_shapes = {
      MakeDynamicXlaInput(variable),
      MakeDynamicXlaInput(variable + 1),
  };
  // The first occurrence may be a broadcast singleton. A + 1 = 241 gives
  // the stronger runtime evidence A = 240.
  const std::vector<TensorShape> runtime_shapes = {
      TensorShape({1, 16}),
      TensorShape({241, 16}),
  };
  const std::vector<int> runtime_input_indices = {0, 1};
  const std::vector<std::string> runtime_input_names = {"singleton", "full"};

  DynamicBatchResolutionResult result = ResolveForTest(
      xla_shapes, runtime_shapes, runtime_input_indices, runtime_input_names);

  EXPECT_TRUE(result.can_run);
  ASSERT_TRUE(result.has_batch_size);
  EXPECT_EQ(result.batch_size, 240);
}

TEST(DynamicRuntimeSolveTest, KeepsConsistentNonSingletonEvidence) {
  const xla::DExpr variable = xla::DExpr::Var(1);
  const std::vector<xla::Shape> xla_shapes = {
      MakeDynamicXlaInput(variable),
      MakeDynamicXlaInput(variable / 3),
  };
  const std::vector<TensorShape> runtime_shapes = {
      TensorShape({720, 16}),
      TensorShape({240, 16}),
  };
  const std::vector<int> runtime_input_indices = {0, 1};
  const std::vector<std::string> runtime_input_names = {"full", "third"};

  DynamicBatchResolutionResult result = ResolveForTest(
      xla_shapes, runtime_shapes, runtime_input_indices, runtime_input_names);

  EXPECT_TRUE(result.can_run);
  ASSERT_TRUE(result.has_batch_size);
  EXPECT_EQ(result.batch_size, 720);
}

TEST(DynamicRuntimeSolveTest, RejectsConflictingNonSingletonEvidence) {
  const std::vector<xla::Shape> xla_shapes = {
      MakeDynamicXlaInput(xla::DExpr::Var(1)),
      MakeDynamicXlaInput(xla::DExpr::Var(1)),
  };
  const std::vector<TensorShape> runtime_shapes = {
      TensorShape({240, 16}),
      TensorShape({720, 16}),
  };
  const std::vector<int> runtime_input_indices = {0, 1};
  const std::vector<std::string> runtime_input_names = {"first", "second"};

  DynamicBatchResolutionResult result = ResolveForTest(
      xla_shapes, runtime_shapes, runtime_input_indices, runtime_input_names);

  EXPECT_FALSE(result.can_run);
  EXPECT_FALSE(result.has_batch_size);
  EXPECT_NE(result.diagnostic.find("conflicting non-singleton candidates"),
            std::string::npos);
}

// Exercise the same recovery policy before padding and at executable entry.
void ExpectRootResolution(const std::vector<xla::DExpr>& expressions,
                          const std::vector<int64_t>& observed,
                          std::optional<int64_t> expected_root) {
  ASSERT_EQ(expressions.size(), observed.size());
  std::vector<XlaArgument> args;
  std::vector<xla::Shape> xla_shapes;
  std::vector<TensorShape> runtime_shapes;
  std::vector<int> mapping;
  std::vector<std::string> names;
  for (int i = 0; i < expressions.size(); ++i) {
    args.push_back(MakeDynamicArgument(observed[i], expressions[i]));
    xla_shapes.push_back(MakeDynamicXlaInput(expressions[i]));
    runtime_shapes.emplace_back(TensorShape({observed[i], 16}));
    mapping.push_back(i);
    names.push_back(std::to_string(i));
  }
  const auto compile = AnalyzeIgnoredDynamicArgumentOccurrences(args);
  const auto runtime = ResolveForTest(xla_shapes, runtime_shapes, mapping, names);
  EXPECT_EQ(compile.can_run, expected_root.has_value()) << compile.diagnostic;
  EXPECT_EQ(runtime.can_run, expected_root.has_value()) << runtime.diagnostic;
  if (expected_root.has_value()) {
    EXPECT_EQ(compile.root_value, expected_root);
    ASSERT_TRUE(runtime.has_batch_size) << runtime.diagnostic;
    EXPECT_EQ(runtime.batch_size, *expected_root);
  } else {
    EXPECT_FALSE(runtime.has_batch_size);
  }
}

TEST(DynamicRootValidationTest, ChecksRoundedDivisionUsingRecoveredRoot) {
  const xla::DExpr a = xla::DExpr::Var(1);
  // 721 / 240 is 3. Inverting that observation as exact division gives 720.
  ExpectRootResolution({a, a / 240}, {721, 3}, 721);
  ExpectRootResolution({a / 240, a}, {3, 721}, 721);
  ExpectRootResolution({a + 1, a / 240}, {722, 3}, 721);
}

TEST(DynamicRootValidationTest, RequiresUnambiguousEvidence) {
  const xla::DExpr a = xla::DExpr::Var(1);
  ExpectRootResolution({a / 240}, {3}, std::nullopt);
  ExpectRootResolution({a / 240 + 1}, {4}, std::nullopt);
  ExpectRootResolution({xla::DExpr::Const(100) / a}, {3}, std::nullopt);
  ExpectRootResolution({(a / 240) * 240}, {720}, std::nullopt);
}

TEST(DynamicRootValidationTest, PreservesInjectiveArithmeticRecovery) {
  const xla::DExpr a = xla::DExpr::Var(1);
  ExpectRootResolution({3 * a, a / 240}, {2163, 3}, 721);
  ExpectRootResolution({xla::DExpr::Const(100) - a}, {90}, 10);
  ExpectRootResolution({xla::DExpr::Max(a - 2, xla::DExpr::Const(0))}, {98}, 100);
  ExpectRootResolution({a / 1}, {721}, 721);
}

TEST(DynamicRootValidationTest, ValidatesExpressionsThatCannotBeInverted) {
  const xla::DExpr a = xla::DExpr::Var(1);
  const xla::DExpr clamped = xla::DExpr::Max(a - 10, xla::DExpr::Const(0));
  ExpectRootResolution({a, clamped}, {5, 0}, 5);
  ExpectRootResolution({a, clamped}, {5, 2}, std::nullopt);
  ExpectRootResolution({a, a * a}, {5, 25}, 5);
  ExpectRootResolution({a, a * a}, {5, 24}, std::nullopt);
  ExpectRootResolution({a, a / 240}, {721, 4}, std::nullopt);
}

TEST(DynamicRootValidationTest, RejectsUnevaluableObservations) {
  const xla::DExpr a = xla::DExpr::Var(1);
  ExpectRootResolution({a, a / (a - 5)}, {5, 2}, std::nullopt);
  ExpectRootResolution({a, a + xla::DExpr::Unknown()}, {5, 2}, std::nullopt);
  ExpectRootResolution({a, xla::DExpr::Var(2)}, {5, 5}, std::nullopt);
}

TEST(DynamicRootValidationTest, KeepsBroadcastExceptionForDividedDimension) {
  const xla::DExpr a = xla::DExpr::Var(1);
  std::vector<XlaArgument> args = {MakeDynamicArgument(720, a),
                                 MakeDynamicArgument(1, a / 240)};
  auto decision = AnalyzeIgnoredDynamicArgumentOccurrences(args);
  ASSERT_TRUE(decision.can_run) << decision.diagnostic;
  EXPECT_EQ(decision.root_value, 720);
  ASSERT_EQ(decision.ignored_occurrences.size(), 1);
  EXPECT_EQ(decision.ignored_occurrences[0].arg_index, 1);
  EXPECT_FALSE(decision.ignored_occurrences[0].solved_value.has_value());
  StripIgnoredDynamicArgumentOccurrences(decision.ignored_occurrences, &args);
  EXPECT_FALSE(std::get<TensorShape>(args[1].shape).get_expression(0));
}

TEST(DynamicRootValidationTest, ChecksConstantContentsAgainstRuntimeRoot) {
  const xla::DExpr a = xla::DExpr::Var(1);
  std::vector<XlaArgument> args = {
      MakeDynamicArgument(721, a),
      MakeConstantContents(DT_INT32, {3}, {a / 240})};
  auto decision = AnalyzeIgnoredDynamicArgumentOccurrences(args);
  ASSERT_TRUE(decision.can_run) << decision.diagnostic;
  EXPECT_EQ(decision.root_value, 721);

  // A value of 1 in a shape tensor is still checked as an ordinary value.
  args[1].constant_value.flat<int32>()(0) = 1;
  EXPECT_FALSE(AnalyzeIgnoredDynamicArgumentOccurrences(args).can_run);
  args.erase(args.begin());
  EXPECT_FALSE(AnalyzeIgnoredDynamicArgumentOccurrences(args).can_run);
}

TEST(DynamicRootValidationTest, RejectsMissingMappingOrWrongRank) {
  const std::vector<xla::Shape> xla_shapes = {
      MakeDynamicXlaInput(xla::DExpr::Var(1))};
  const std::vector<TensorShape> runtime_shapes = {TensorShape({5, 16})};
  const std::vector<int> missing = {-1};
  const std::vector<int> mapped = {0};
  const std::vector<std::string> names = {"input"};
  EXPECT_FALSE(ResolveForTest(xla_shapes, runtime_shapes, missing, names).can_run);
  const std::vector<TensorShape> wrong_rank = {TensorShape({5})};
  EXPECT_FALSE(ResolveForTest(xla_shapes, wrong_rank, mapped, names).can_run);
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

TEST(DynamicSolveFilterTest,
     IgnoresSingletonAcrossExpressionsOfTheSameVariable) {
  const xla::DExpr variable = xla::DExpr::Var(1);
  std::vector<XlaArgument> args = {
      MakeDynamicArgument(1, variable),
      MakeDynamicArgument(241, variable + 1),
  };

  DynamicSolveFilterDecision decision =
      AnalyzeIgnoredDynamicArgumentOccurrences(args);

  ASSERT_TRUE(decision.can_run);
  ASSERT_EQ(decision.ignored_occurrences.size(), 1);
  EXPECT_EQ(decision.ignored_occurrences[0].arg_index, 0);
  EXPECT_EQ(decision.ignored_occurrences[0].observed_value, 1);
  EXPECT_EQ(decision.ignored_occurrences[0].solved_value, 1);
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

TEST(DynamicSolveFilterTest,
     DoesNotIgnoreSingletonConstantValueEvidence) {
  std::vector<XlaArgument> args = {
      MakeDynamicConstantArgument(1),
      MakeDynamicArgument(240),
  };

  DynamicSolveFilterDecision decision =
      AnalyzeIgnoredDynamicArgumentOccurrences(args);

  EXPECT_FALSE(decision.can_run);
  EXPECT_TRUE(decision.ignored_occurrences.empty());
  EXPECT_NE(decision.diagnostic.find("dynamic observation mismatch"),
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

TEST(DynamicSolveFilterTest, StaticArgumentsKeepConcreteRuntimeShapes) {
  XlaArgument dynamic_arg = MakeDynamicArgument(37);
  dynamic_arg.constant_value_expressions.resize(1);
  dynamic_arg.constant_value_expressions[0].set_variable_id(1);

  std::vector<XlaArgument> args = {dynamic_arg};
  std::vector<XlaArgument> static_args =
      BuildStaticCompilationArguments(args);

  ASSERT_EQ(static_args.size(), 1);
  const TensorShape& static_shape =
      std::get<TensorShape>(static_args[0].shape);
  EXPECT_EQ(static_shape.dim_size(0), 37);
  EXPECT_EQ(static_shape.dim_size(1), 16);
  EXPECT_TRUE(static_shape.get_expressions().empty());
  EXPECT_TRUE(static_args[0].constant_value_expressions.empty());
}

TEST(DynamicSolveFilterTest, StaticArgumentsClearXlaShapeDynamism) {
  xla::Shape dynamic_shape;
  dynamic_shape.set_element_type(xla::F32);
  dynamic_shape.add_dimensions(37, /*is_dynamic=*/true,
                               xla::DExpr::Var(1));

  XlaArgument dynamic_arg;
  dynamic_arg.kind = XlaArgument::kParameter;
  dynamic_arg.type = DT_FLOAT;
  dynamic_arg.shape = dynamic_shape;
  std::vector<XlaArgument> args = {dynamic_arg};

  std::vector<XlaArgument> static_args =
      BuildStaticCompilationArguments(args);

  const xla::Shape& static_shape = std::get<xla::Shape>(static_args[0].shape);
  EXPECT_EQ(static_shape.dimensions(0), 37);
  EXPECT_FALSE(static_shape.is_dynamic_dimension(0));
  ASSERT_EQ(static_shape.expressions().size(), 1);
  EXPECT_TRUE(static_shape.expressions(0)->is_constant());
  EXPECT_EQ(static_shape.expressions(0)->get_val(), 37);
}

}  // namespace
}  // namespace xla_ops_internal
}  // namespace tensorflow
