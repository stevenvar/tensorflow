/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/kernels/xla_ops_internal.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/device_compilation_profiler.h"
#include "tensorflow/compiler/jit/device_compiler.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/pjrt_compile_util.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/jit/xla_compiler_options_util.h"
#include "tensorflow/compiler/jit/xla_host_recv_device_context.h"
#include "tensorflow/compiler/jit/xla_host_send_device_context.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/jit/xla_batch_matcher.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/printer.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/shape_expr.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/batch_size_resource.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tsl/platform/statusor.h"

// OP_REQUIRES_OK_RETURN is the same as OP_REQUIRES_OK except that
// in error case, it returns RET instead of void.
#define OP_REQUIRES_OK_RETURN(CTX, RET, ...)                \
  do {                                                      \
    ::tensorflow::Status _s(__VA_ARGS__);                   \
    if (!TF_PREDICT_TRUE(_s.ok())) {                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      return RET;                                           \
    }                                                       \
  } while (0)

namespace tensorflow {

namespace {
constexpr char kUserInferredValueContentsAttrName[] =
    "_user_inferred_value_contents";
using XlaDeviceCompiler =
    DeviceCompiler<xla::LocalExecutable, xla::LocalClient>;
using PjRtDeviceCompiler =
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>;

auto* xla_launch_counter = monitoring::Counter<1>::New(
    "/tensorflow/core/xla_launch_counter",
    "The number of times a XlaLaunch is called.", "device");

// A closure describing how to run a compiled version of a TensorFlow function.
//
// It may seem unusual to stick the resource variable snapshots in this class.
// This is necessary: we need to use the snapshots observed by the compiler as
// the initial values for the resource variables (and cannot snapshot them again
// during execution) because otherwise we risk observing a different snapshot
// with shapes different from what we compiled for.
template <typename ExecutableType, typename ClientType>
class ExecutableClosure {
 public:
  explicit ExecutableClosure(
      ClientType* client, ExecutableType* executable,
      const XlaCompiler::CompilationResult* compilation_result,
      ResourceVarsSnapshot resource_var_snapshots, int num_constant_args)
      : client_(client),
        executable_(executable),
        compilation_result_(compilation_result),
        resource_var_snapshots_(std::move(resource_var_snapshots)),
        num_constant_args_(num_constant_args) {}

  ExecutableClosure(ExecutableClosure&&) = default;
  ExecutableClosure& operator=(ExecutableClosure&&) = default;

  ClientType* client() const { return client_; }
  ExecutableType* executable() const { return executable_; }
  const XlaCompiler::CompilationResult* compilation_result() const {
    return compilation_result_;
  }
  const ResourceVarsSnapshot& resource_var_snapshots() const {
    return resource_var_snapshots_;
  }
  int num_constant_args() const { return num_constant_args_; }

 private:
  ClientType* client_;
  ExecutableType* executable_;
  const XlaCompiler::CompilationResult* compilation_result_;
  ResourceVarsSnapshot resource_var_snapshots_;
  int num_constant_args_;

  ExecutableClosure(const ExecutableClosure&) = delete;
  void operator=(const ExecutableClosure&) = delete;
};

// This maintains a mapping from a globally unique ID to ExecutableClosure
// instances.
template <typename ExecutableType, typename ClientType>
class ExecutableClosureStore {
 public:
  ExecutableClosureStore() : key_counter_(0) {}

  using KeyT = string;

  KeyT Produce(ExecutableClosure<ExecutableType, ClientType> result) {
    mutex_lock l(mutex_);
    KeyT key = absl::StrCat(key_counter_++);
    bool insert_successful = closures_.emplace(key, std::move(result)).second;
    DCHECK(insert_successful);
    (void)insert_successful;
    return key;
  }

  ExecutableClosure<ExecutableType, ClientType> Consume(const KeyT& key) {
    mutex_lock l(mutex_);
    auto it = closures_.find(key);
    DCHECK(it != closures_.end());
    ExecutableClosure<ExecutableType, ClientType> value = std::move(it->second);
    closures_.erase(it);
    return value;
  }

  static ExecutableClosureStore* Global() {
    static ExecutableClosureStore* instance = new ExecutableClosureStore;
    return instance;
  }

 private:
  mutex mutex_;
  int64_t key_counter_ TF_GUARDED_BY(mutex_);
  absl::flat_hash_map<KeyT, ExecutableClosure<ExecutableType, ClientType>>
      closures_ TF_GUARDED_BY(mutex_);

  ExecutableClosureStore(const ExecutableClosureStore&) = delete;
  void operator=(const ExecutableClosureStore&) = delete;
};

using XlaExecutableClosure =
    ExecutableClosure<xla::LocalExecutable, xla::LocalClient>;
using XlaExecutableClosureStore =
    ExecutableClosureStore<xla::LocalExecutable, xla::LocalClient>;
using PjRtExecutableClosure =
    ExecutableClosure<xla::PjRtLoadedExecutable, xla::PjRtClient>;
using PjRtExecutableClosureStore =
    ExecutableClosureStore<xla::PjRtLoadedExecutable, xla::PjRtClient>;

struct DynamicBatchResolutionResult {
  bool can_run = true;
  bool has_batch_size = false;
  int64_t batch_size = 0;
  bool batch_size_resource_not_found = false;
  std::string diagnostic;
};

struct DynamicSolveCandidate {
  int64_t solved_value;
  int64_t observed_dim_value;
  std::string context;
};

}  // namespace

namespace xla_ops_internal {

std::optional<int64_t> GetConstantArgumentElementValue(
    const XlaArgument& arg, int index) {
  if (index < 0 || index >= arg.constant_value.NumElements()) {
    return std::nullopt;
  }
  switch (arg.constant_value.dtype()) {
    case DT_INT32:
      return static_cast<int64_t>(arg.constant_value.flat<int32>()(index));
    case DT_INT64:
      return static_cast<int64_t>(arg.constant_value.flat<int64_t>()(index));
    default:
      return std::nullopt;
  }
}

void SetConstantArgumentExpressionToLiteralValue(XlaArgument* arg,
                                                 int index) {
  if (arg == nullptr || index < 0 ||
      index >= arg->constant_value_expressions.size()) {
    return;
  }
  std::optional<int64_t> value = GetConstantArgumentElementValue(*arg, index);
  if (!value.has_value()) {
    arg->constant_value_expressions[index].Clear();
    return;
  }
  arg->constant_value_expressions[index].Clear();
  arg->constant_value_expressions[index].set_constant_value(*value);
}

DynamicSolveFilterDecision AnalyzeIgnoredDynamicArgumentOccurrences(
    absl::Span<const XlaArgument> args) {
  struct Candidate {
    int64_t solved_value;
    int64_t observed_value;
    IgnoredDynamicArgumentOccurrence occurrence;
  };

  DynamicSolveFilterDecision result;
  std::map<std::string, std::vector<Candidate>> expr_to_candidates;

  for (int arg_index = 0; arg_index < args.size(); ++arg_index) {
    const XlaArgument& arg = args[arg_index];
    if (absl::holds_alternative<TensorShape>(arg.shape)) {
      const TensorShape& shape = std::get<TensorShape>(arg.shape);
      for (int dim = 0; dim < shape.get_expressions().size(); ++dim) {
        const xla::DExpr& expr = shape.get_expression(dim);
        if (!(expr && expr->is_dynamic())) {
          continue;
        }
        xla::DExpr simplified_expr = expr.simplify();
        std::optional<int64_t> solved_value =
            simplified_expr->solve(shape.dim_size(dim));
        if (!solved_value.has_value()) {
          continue;
        }
        const std::string expr_string = DExprToString(simplified_expr);
        expr_to_candidates[expr_string].push_back(Candidate{
            *solved_value,
            shape.dim_size(dim),
            IgnoredDynamicArgumentOccurrence{
                IgnoredDynamicArgumentOccurrence::Source::kShapeDimension,
                arg_index,
                dim,
                shape.dim_size(dim),
                *solved_value,
                expr_string,
            },
        });
      }
    }

    if (arg.kind != XlaCompiler::Argument::kConstant ||
        arg.constant_value_expressions.empty()) {
      continue;
    }
    for (int element_index = 0;
         element_index < arg.constant_value_expressions.size();
         ++element_index) {
      xla::DExpr expr =
          xla::DExprFromProto(arg.constant_value_expressions[element_index]);
      if (!(expr && expr->is_dynamic())) {
        continue;
      }
      std::optional<int64_t> observed_value =
          GetConstantArgumentElementValue(arg, element_index);
      if (!observed_value.has_value()) {
        continue;
      }
      xla::DExpr simplified_expr = expr.simplify();
      std::optional<int64_t> solved_value =
          simplified_expr->solve(*observed_value);
      if (!solved_value.has_value()) {
        continue;
      }
      const std::string expr_string = DExprToString(simplified_expr);
      expr_to_candidates[expr_string].push_back(Candidate{
          *solved_value,
          *observed_value,
          IgnoredDynamicArgumentOccurrence{
              IgnoredDynamicArgumentOccurrence::Source::kConstantValueElement,
              arg_index,
              element_index,
              *observed_value,
              *solved_value,
              expr_string,
          },
      });
    }
  }

  std::vector<std::string> expr_summaries;
  for (const auto& [expr, candidates] : expr_to_candidates) {
    std::set<int64_t> singleton_values;
    std::set<int64_t> nonsingleton_values;
    for (const auto& candidate : candidates) {
      // A singleton occurrence can legally be an implicitly broadcast operand.
      // Prefer an agreeing non-singleton occurrence of the same expression.
      if (candidate.observed_value == 1) {
        singleton_values.insert(candidate.solved_value);
      } else {
        nonsingleton_values.insert(candidate.solved_value);
      }
    }

    std::optional<int64_t> chosen_value;
    if (!nonsingleton_values.empty()) {
      if (nonsingleton_values.size() != 1) {
        result.can_run = false;
        expr_summaries.push_back(absl::StrCat(
            "expr=", expr, " reason=conflicting non-singleton candidates",
            " values={", absl::StrJoin(nonsingleton_values, ", "), "}"));
        continue;
      }
      chosen_value = *nonsingleton_values.begin();
      for (const auto& candidate : candidates) {
        if (candidate.observed_value == 1 &&
            candidate.solved_value != *chosen_value) {
          LOG(INFO) << "Ignoring singleton-derived dynamic "
                    << "occurrence during XLA signature filtering: expr="
                    << expr
                    << " chosen_value=" << *chosen_value
                    << " ignored_observed_value=" << candidate.observed_value
                    << " ignored_solved_value=" << candidate.solved_value
                    << " arg_index=" << candidate.occurrence.arg_index
                    << (candidate.occurrence.source ==
                                IgnoredDynamicArgumentOccurrence::Source::
                                    kShapeDimension
                            ? " dim="
                            : " element=")
                    << candidate.occurrence.dim_or_index;
          result.ignored_occurrences.push_back(candidate.occurrence);
        }
      }
    } else if (!singleton_values.empty()) {
      if (singleton_values.size() != 1) {
        result.can_run = false;
        expr_summaries.push_back(absl::StrCat(
            "expr=", expr,
            " reason=conflicting singleton-derived candidates",
            " values={", absl::StrJoin(singleton_values, ", "), "}"));
        continue;
      }
      chosen_value = *singleton_values.begin();
    }

    expr_summaries.push_back(absl::StrCat(
        "expr=", expr, " chosen=",
        chosen_value.has_value() ? std::to_string(*chosen_value)
                                 : std::string("<none>"),
        " singleton_derived_values={",
        absl::StrJoin(singleton_values, ", "), "} nonsingleton_values={",
        absl::StrJoin(nonsingleton_values, ", "),
        "}"));
  }

  if (!result.can_run) {
    result.diagnostic = absl::StrCat(
        "Failed to recover a unique XLA dynamic batch size after solve-time "
        "candidate filtering. solved_expressions=[",
        absl::StrJoin(expr_summaries, " | "), "]");
    return result;
  }

  if (!result.ignored_occurrences.empty()) {
    std::vector<std::string> ignored_summaries;
    ignored_summaries.reserve(result.ignored_occurrences.size());
    for (const auto& occurrence : result.ignored_occurrences) {
      ignored_summaries.push_back(absl::StrCat(
          occurrence.source ==
                  IgnoredDynamicArgumentOccurrence::Source::kShapeDimension
              ? "shape_arg="
              : "const_arg=",
          occurrence.arg_index,
          occurrence.source ==
                  IgnoredDynamicArgumentOccurrence::Source::kShapeDimension
              ? " dim="
              : " element=",
          occurrence.dim_or_index, " expr=", occurrence.expr,
          " observed=", occurrence.observed_value,
          " solved=", occurrence.solved_value));
    }
    result.diagnostic = absl::StrCat(
        "Ignoring singleton-derived dynamic occurrences for XLA "
        "signature/HLO: ",
        absl::StrJoin(ignored_summaries, "; "));
  }
  return result;
}

void StripIgnoredDynamicArgumentOccurrences(
    const std::vector<IgnoredDynamicArgumentOccurrence>& ignored_occurrences,
    std::vector<XlaArgument>* args) {
  if (args == nullptr) {
    return;
  }
  for (const auto& occurrence : ignored_occurrences) {
    if (occurrence.arg_index < 0 || occurrence.arg_index >= args->size()) {
      continue;
    }
    XlaArgument& arg = (*args)[occurrence.arg_index];
    if (occurrence.source ==
        IgnoredDynamicArgumentOccurrence::Source::kShapeDimension) {
      if (!absl::holds_alternative<TensorShape>(arg.shape)) {
        continue;
      }
      LOG(INFO) << "Dropping ignored dynamic shape expression from XLA "
                << "signature/HLO: arg_index=" << occurrence.arg_index
                << " dim=" << occurrence.dim_or_index
                << " expr=" << occurrence.expr
                << " observed_value=" << occurrence.observed_value
                << " solved_value=" << occurrence.solved_value;
      TensorShape& shape = std::get<TensorShape>(arg.shape);
      shape.set_expression(occurrence.dim_or_index, xla::DExpr());
      continue;
    }

    LOG(INFO) << "Dropping ignored dynamic constant-value expression from XLA "
              << "signature/HLO: arg_index=" << occurrence.arg_index
              << " element=" << occurrence.dim_or_index
              << " expr=" << occurrence.expr
              << " observed_value=" << occurrence.observed_value
              << " solved_value=" << occurrence.solved_value;
    SetConstantArgumentExpressionToLiteralValue(&arg, occurrence.dim_or_index);
  }
}

}  // namespace xla_ops_internal

namespace {

using xla_ops_internal::AnalyzeIgnoredDynamicArgumentOccurrences;
using xla_ops_internal::DynamicSolveFilterDecision;
using xla_ops_internal::StripIgnoredDynamicArgumentOccurrences;

DynamicBatchResolutionResult ResolveDynamicBatchSizeFromRuntimeInputs(
    OpKernelContext* ctx, const XlaCompiler::CompilationResult& comp_result,
    int num_constant_args, bool log_solves, absl::string_view cluster_name,
    const NodeDef& op_def) {
  DynamicBatchResolutionResult result;
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  if (!flags->tf_xla_enable_dynamic_sizes) {
    return result;
  }
  const bool enable_dynamic_solve_majority_vote =
      flags->tf_xla_enable_dynamic_solve_majority_vote;

  auto resolve_runtime_input_index = [&](int xla_input_index) -> int {
    const bool has_runtime_key =
        ctx->num_inputs() > 0 &&
        ctx->input_dtype(ctx->num_inputs() - 1) == DT_STRING &&
        ctx->num_inputs() == comp_result.xla_input_shapes.size() + 1;
    if (has_runtime_key) {
      return xla_input_index;
    }
    return num_constant_args + xla_input_index;
  };

  std::set<int64_t> dyn_vals;
  std::map<std::string, std::vector<DynamicSolveCandidate>> expr_to_candidates;

  for (int i = 0; i < comp_result.xla_input_shapes.size(); ++i) {
    const auto& xla_shape = comp_result.xla_input_shapes[i];
    const bool has_input_mapping = i < comp_result.input_mapping.size();
    const int input_idx = resolve_runtime_input_index(i);
    if (!xla_shape.IsArray() || xla_shape.expressions().empty()) {
      continue;
    }

    for (int dim = 0; dim < xla_shape.expressions().size(); ++dim) {
      const auto& expr = xla_shape.expressions(dim);
      if (!(expr && expr->is_dynamic())) {
        continue;
      }
      xla::DExpr simplified_expr = expr.simplify();
      if (!has_input_mapping) {
        continue;
      }
      if (input_idx < 0 || input_idx >= ctx->num_inputs()) {
        continue;
      }
      const bool is_runtime_key_input =
          ctx->input_dtype(input_idx) == DT_STRING &&
          input_idx == op_def.input_size() - 1;
      const std::string runtime_input_name =
          input_idx < op_def.input_size() ? op_def.input(input_idx)
                                          : std::string("<unknown>");
      if (is_runtime_key_input) {
        if (log_solves) {
          VLOG(1) << "Skipping dynamic expression solve for runtime key input "
                  << "cluster=" << cluster_name
                  << " xla_input_index=" << i
                  << " runtime_input_index=" << input_idx
                  << " input_name=" << runtime_input_name;
        }
        continue;
      }
      int64_t size = ctx->input(input_idx).shape().dim_size(dim);
      std::optional<int64_t> dyn_val = simplified_expr->solve(size);
      if (log_solves) {
        LOG(INFO) << "Dynamic solve: cluster=" << cluster_name
                  << " runtime_input_index=" << input_idx
                  << " xla_input_index=" << i << " dim=" << dim
                  << " input_name=" << runtime_input_name
                  << " expr=" << DExprToString(simplified_expr)
                  << " target_size=" << size << " result="
                  << (dyn_val.has_value() ? std::to_string(*dyn_val)
                                          : std::string("<none>"));
      }
      if (!dyn_val.has_value()) {
        if (log_solves) {
          xla::StringPrinter printer;
          simplified_expr->print(&printer);
          LOG(INFO) << "Failed to solve dynamic expression: "
                    << "xla_input_index=" << i
                    << " runtime_input_index=" << input_idx
                    << " input_name=" << runtime_input_name
                    << " dim=" << dim
                    << " runtime_dim_size=" << size
                    << " expr=" << std::move(printer).ToString();
        }
        continue;
      }
      const std::string expr_string = DExprToString(simplified_expr);
      const std::string context = absl::StrCat(
          "xla_input_index=", i, " runtime_input_index=", input_idx,
          " input_name=", runtime_input_name, " dim=", dim,
          " runtime_dim_size=", size,
          " solved_dynamic_value=", *dyn_val);
      expr_to_candidates[expr_string].push_back(
          DynamicSolveCandidate{*dyn_val, size, context});
    }
  }

  std::vector<std::string> expr_summaries;
  std::optional<std::set<int>> expected_dyn_ids;
  bool mismatched_dyn_ids = false;
  bool candidate_filter_rejected = false;
  bool majority_vote_used = false;
  bool ambiguous_majority_vote = false;
  std::vector<std::string> voted_expr_summaries;
  for (const auto& [expr, candidates] : expr_to_candidates) {
    std::set<int64_t> singleton_values;
    std::set<int64_t> nonsingleton_values;
    std::vector<std::string> singleton_contexts;
    std::vector<std::string> nonsingleton_contexts;
    for (const auto& candidate : candidates) {
      // Runtime {1} is compatible with an implicit broadcast. It is weaker
      // evidence than a consistent non-singleton observation.
      if (candidate.observed_dim_value == 1) {
        singleton_values.insert(candidate.solved_value);
        singleton_contexts.push_back(candidate.context);
      } else {
        nonsingleton_values.insert(candidate.solved_value);
        nonsingleton_contexts.push_back(candidate.context);
      }
    }

    std::optional<int64_t> chosen_value;
    std::string filter_reason;
    if (!nonsingleton_values.empty()) {
      if (nonsingleton_values.size() == 1) {
        chosen_value = *nonsingleton_values.begin();
        filter_reason =
            singleton_values.empty()
                ? "used non-singleton evidence"
                : "preferred non-singleton evidence over singleton-derived "
                  "candidates";
      } else {
        if (!enable_dynamic_solve_majority_vote) {
          candidate_filter_rejected = true;
          filter_reason =
              "conflicting non-singleton candidates; majority vote disabled "
              "for debugging";
          LOG(INFO) << "Dynamic solve encountered conflicting non-singleton "
                       "candidates for expr="
                    << expr
                    << " but majority vote is disabled for debugging.";
        } else {
        std::map<int64_t, std::vector<std::string>> vote_contexts;
        for (const auto& candidate : candidates) {
          vote_contexts[candidate.solved_value].push_back(candidate.context);
        }
        int64_t winner_value = 0;
        size_t winner_count = 0;
        size_t runner_up_count = 0;
        bool have_winner = false;
        bool tie_for_winner = false;
        std::vector<std::string> tally_parts;
        for (const auto& [value, contexts] : vote_contexts) {
          tally_parts.push_back(absl::StrCat(
              value, " x", contexts.size(), " [",
              absl::StrJoin(contexts, "; "), "]"));
          if (!have_winner || contexts.size() > winner_count) {
            runner_up_count = have_winner ? winner_count : 0;
            winner_value = value;
            winner_count = contexts.size();
            have_winner = true;
            tie_for_winner = false;
          } else if (contexts.size() == winner_count) {
            tie_for_winner = true;
          } else if (contexts.size() > runner_up_count) {
            runner_up_count = contexts.size();
          }
        }
        const bool clear_majority = !tie_for_winner && winner_count >= 2 &&
                                    winner_count > runner_up_count;
        if (clear_majority) {
          chosen_value = winner_value;
          majority_vote_used = true;
          filter_reason =
              "singleton filtering conflicted; majority vote kept value";
          std::vector<std::string> dropped_parts;
          for (const auto& [value, contexts] : vote_contexts) {
            if (value == winner_value) continue;
            dropped_parts.push_back(
                absl::StrCat(value, " x", contexts.size()));
          }
          LOG(INFO) << "Dynamic solve majority vote kept value for expr="
                    << expr << " kept=" << winner_value
                    << " kept_count=" << winner_count
                    << " runner_up_count=" << runner_up_count
                    << " dropped=["
                    << (dropped_parts.empty() ? std::string()
                                              : absl::StrJoin(dropped_parts, ", "))
                    << "] tallies=[" << absl::StrJoin(tally_parts, " | ")
                    << "]";
          voted_expr_summaries.push_back(absl::StrCat(
              "expr=", expr, " kept=", winner_value, " count=", winner_count,
              " dropped=[",
              dropped_parts.empty() ? std::string()
                                    : absl::StrJoin(dropped_parts, ", "),
              "]"));
        } else {
          candidate_filter_rejected = true;
          ambiguous_majority_vote = true;
          filter_reason =
              "conflicting non-singleton candidates and no clear majority "
              "vote";
          LOG(INFO) << "Dynamic solve majority vote could not choose a unique "
                       "value for expr="
                    << expr << " tallies=[" << absl::StrJoin(tally_parts, " | ")
                    << "]";
        }
        }
      }
    } else if (!singleton_values.empty()) {
      if (singleton_values.size() == 1) {
        chosen_value = *singleton_values.begin();
        filter_reason = "used singleton-derived evidence only";
      } else {
        if (!enable_dynamic_solve_majority_vote) {
          candidate_filter_rejected = true;
          filter_reason =
              "conflicting singleton-derived candidates; majority "
              "vote disabled for debugging";
          LOG(INFO) << "Dynamic solve encountered conflicting "
                       "singleton-derived candidates for expr="
                    << expr
                    << " but majority vote is disabled for debugging.";
        } else {
        std::map<int64_t, std::vector<std::string>> vote_contexts;
        for (const auto& candidate : candidates) {
          vote_contexts[candidate.solved_value].push_back(candidate.context);
        }
        int64_t winner_value = 0;
        size_t winner_count = 0;
        size_t runner_up_count = 0;
        bool have_winner = false;
        bool tie_for_winner = false;
        std::vector<std::string> tally_parts;
        for (const auto& [value, contexts] : vote_contexts) {
          tally_parts.push_back(absl::StrCat(
              value, " x", contexts.size(), " [",
              absl::StrJoin(contexts, "; "), "]"));
          if (!have_winner || contexts.size() > winner_count) {
            runner_up_count = have_winner ? winner_count : 0;
            winner_value = value;
            winner_count = contexts.size();
            have_winner = true;
            tie_for_winner = false;
          } else if (contexts.size() == winner_count) {
            tie_for_winner = true;
          } else if (contexts.size() > runner_up_count) {
            runner_up_count = contexts.size();
          }
        }
        const bool clear_majority = !tie_for_winner && winner_count >= 2 &&
                                    winner_count > runner_up_count;
        if (clear_majority) {
          chosen_value = winner_value;
          majority_vote_used = true;
          filter_reason =
              "singleton-only evidence conflicted; majority vote kept value";
          std::vector<std::string> dropped_parts;
          for (const auto& [value, contexts] : vote_contexts) {
            if (value == winner_value) continue;
            dropped_parts.push_back(
                absl::StrCat(value, " x", contexts.size()));
          }
          LOG(INFO) << "Dynamic solve majority vote kept value for expr="
                    << expr << " kept=" << winner_value
                    << " kept_count=" << winner_count
                    << " runner_up_count=" << runner_up_count
                    << " dropped=["
                    << (dropped_parts.empty() ? std::string()
                                              : absl::StrJoin(dropped_parts, ", "))
                    << "] tallies=[" << absl::StrJoin(tally_parts, " | ")
                    << "]";
          voted_expr_summaries.push_back(absl::StrCat(
              "expr=", expr, " kept=", winner_value, " count=", winner_count,
              " dropped=[",
              dropped_parts.empty() ? std::string()
                                    : absl::StrJoin(dropped_parts, ", "),
              "]"));
        } else {
          candidate_filter_rejected = true;
          ambiguous_majority_vote = true;
          filter_reason =
              "conflicting singleton-derived candidates and no clear "
              "majority vote";
          LOG(INFO) << "Dynamic solve majority vote could not choose a unique "
                       "value for expr="
                    << expr << " tallies=[" << absl::StrJoin(tally_parts, " | ")
                    << "]";
        }
        }
      }
    } else {
      filter_reason = "no dynamic values were solved";
    }

    if (chosen_value.has_value()) {
      dyn_vals.insert(*chosen_value);
    }

    expr_summaries.push_back(absl::StrCat(
        "expr=", expr, " chosen=",
        chosen_value.has_value() ? std::to_string(*chosen_value)
                                 : std::string("<none>"),
        " singleton_derived_values={",
        absl::StrJoin(singleton_values, ", "), "} nonsingleton_values={",
        absl::StrJoin(nonsingleton_values, ", "), "} reason=", filter_reason,
        " singleton_derived_contexts=[",
        absl::StrJoin(singleton_contexts, "; "), "] nonsingleton_contexts=[",
        absl::StrJoin(nonsingleton_contexts, "; "), "]"));
  }
  for (int i = 0; i < comp_result.xla_input_shapes.size(); ++i) {
    const auto& xla_shape = comp_result.xla_input_shapes[i];
    if (!xla_shape.IsArray() || xla_shape.expressions().empty()) continue;
    for (int dim = 0; dim < xla_shape.expressions().size(); ++dim) {
      const auto& expr = xla_shape.expressions(dim);
      if (!(expr && expr->is_dynamic())) {
        continue;
      }
      std::set<int> ids = expr->get_all_ids();
      if (!expected_dyn_ids.has_value()) {
        expected_dyn_ids = ids;
      } else if (*expected_dyn_ids != ids) {
        mismatched_dyn_ids = true;
      }
    }
  }

  if (dyn_vals.size() == 1 && expected_dyn_ids.has_value() &&
      expected_dyn_ids->size() == 1 && !mismatched_dyn_ids) {
    result.has_batch_size = true;
    result.batch_size = *dyn_vals.begin();
    result.diagnostic = absl::StrCat(
        "shared variable ids={", absl::StrJoin(*expected_dyn_ids, ", "),
        "} value=", result.batch_size);
    return result;
  }

  if (expr_to_candidates.empty()) {
    result.diagnostic = "no dynamic values were solved";
  } else if (candidate_filter_rejected) {
    result.can_run = false;
    result.diagnostic = absl::StrCat(
        "Failed to recover a unique XLA dynamic batch size after solve-time "
        "candidate filtering. solved_expressions=[",
        absl::StrJoin(expr_summaries, " | "), "] voted_expressions=[",
        absl::StrJoin(voted_expr_summaries, " | "), "] all_values={",
        absl::StrJoin(dyn_vals, ", "), "}",
        ambiguous_majority_vote ? " reason=no clear majority vote" : "");
    return result;
  } else if (dyn_vals.size() == 1 && mismatched_dyn_ids) {
    result.diagnostic = absl::StrCat(
        "solved dynamic expressions do not share the same variable ids: ",
        absl::StrJoin(expr_summaries, " | "), " voted_expressions=[",
        absl::StrJoin(voted_expr_summaries, " | "), "]");
  } else if (dyn_vals.size() == 1) {
    result.diagnostic = absl::StrCat(
        "solved dynamic expressions do not map to exactly one shared dynamic "
        "variable id: ",
        absl::StrJoin(expr_summaries, " | "), " voted_expressions=[",
        absl::StrJoin(voted_expr_summaries, " | "), "]");
  } else {
    result.can_run = false;
    result.diagnostic = absl::StrCat(
        "Failed to recover a unique XLA dynamic batch size from runtime "
        "input expressions. solved_expressions=[",
        absl::StrJoin(expr_summaries, " | "), "] voted_expressions=[",
        absl::StrJoin(voted_expr_summaries, " | "), "] all_values={",
        absl::StrJoin(dyn_vals, ", "), "}",
        majority_vote_used ? " after majority-vote filtering" : "");
    return result;
  }

  BatchSizeResource* bsr = nullptr;
  ScopedStepContainer* step_container = ctx->step_container();
  absl::Status st = step_container->Lookup<BatchSizeResource>(
      ctx->resource_manager(), BatchSizeResourceName, &bsr);

  if (st.ok()) {
    CHECK(bsr != nullptr);
    result.has_batch_size = true;
    result.batch_size = bsr->GetBatchSize();
    result.diagnostic = absl::StrCat(
        "BatchSizeResource fallback value=", result.batch_size);
    bsr->Unref();
  } else if (IsNotFound(st)) {
    result.batch_size_resource_not_found = true;
  } else {
    result.can_run = false;
    result.diagnostic = st.ToString();
  }

  return result;
}

se::Stream* GetStream(OpKernelContext* ctx) {
  return ctx->op_device_context() ? ctx->op_device_context()->stream()
                                  : nullptr;
}

XlaComputationLaunchContext GetLaunchContext(
    const XlaPlatformInfo& platform_info, OpKernelContext* ctx,
    xla::LocalClient* client, se::DeviceMemoryAllocator* allocator) {
  se::Stream* stream = GetStream(ctx);
  int device_ordinal = stream ? stream->parent()->device_ordinal()
                              : client->default_device_ordinal();
  XlaComputationLaunchContext launch_context(
      client, allocator, device_ordinal,
      /*allocate_xla_tensors=*/platform_info.is_on_xla_device(),
      /*use_multiple_streams=*/platform_info.UseMultipleStreams());
  return launch_context;
}

absl::Status GetTaskName(const absl::string_view device_name,
                         std::string* task_name) {
  string ignored;
  if (!DeviceNameUtils::SplitDeviceName(device_name, task_name, &ignored)) {
    return errors::InvalidArgument("Unable to parse device name: ",
                                   device_name);
  }

  return absl::OkStatus();
}

// Provide SendDeviceMemoryFunction for XLA host callbacks.  This callback
// handles transferring from device to host.
xla::SendDeviceMemoryFunction GetSendDeviceMemoryFunction(
    OpKernelContext* ctx, const std::string& program_key) {
  return
      [ctx, program_key](
          int64_t channel_id, se::Stream* stream, const xla::Shape& shape,
          const se::DeviceMemoryBase& device_memory_base,
          const absl::flat_hash_map<std::string, std::string>& frontend_attrs)
          -> absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<se::Event>>> {
        auto iter = frontend_attrs.find("_xla_host_transfer_rendezvous");

        // Generate the Rendezvous key.
        const std::string& rendezvous_key_base =
            absl::StrCat(program_key, iter->second);

        const std::string& src_device = ctx->device()->name();
        std::string task_prefix;
        TF_RETURN_IF_ERROR(GetTaskName(src_device, &task_prefix));
        const std::string dst_device =
            absl::StrCat(task_prefix, "/device:CPU:0");
        const std::string& rendezvous_key =
            Rendezvous::CreateKey(src_device, /*src_incarnation=*/1, dst_device,
                                  rendezvous_key_base, FrameAndIter(0, 0));
        VLOG(2) << "Rendezvous Key for receiving at host: " << rendezvous_key;

        RendezvousInterface::ParsedKey parsed_key;
        TF_RETURN_IF_ERROR(Rendezvous::ParseKey(rendezvous_key, &parsed_key));

        TF_ASSIGN_OR_RETURN(auto event, stream->parent()->CreateEvent());
        tsl::AsyncValueRef<std::unique_ptr<se::Event>> done_event =
            tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
                std::move(event));

        Rendezvous::Args args;
        // Rendezvous::Args owns the device context pointer.
        args.device_context = new XlaHostRecvDeviceContext(
            stream, device_memory_base, shape, done_event);

        Tensor host_tensor;
        TF_RETURN_IF_ERROR(
            ctx->rendezvous()->Send(parsed_key, args, host_tensor, false));

        return std::move(done_event);
      };
}

// Provide RecvDeviceMemoryFunction for XLA host callbacks.  This callback
// handles transferring from host to device.
xla::RecvDeviceMemoryFunction GetRecvDeviceMemoryFunction(
    OpKernelContext* ctx, const std::string& program_key) {
  return
      [ctx, program_key](
          int64_t channel_id, se::Stream* stream, const xla::Shape& shape,
          se::DeviceMemoryBase* device_memory_base,
          const absl::flat_hash_map<std::string, std::string>& frontend_attrs)
          -> absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<se::Event>>> {
        auto iter = frontend_attrs.find("_xla_host_transfer_rendezvous");

        // Generate the Rendezvous key.
        const std::string& rendezvous_key_base =
            absl::StrCat(program_key, iter->second);

        const std::string& dst_device = ctx->device()->name();
        std::string task_prefix;
        TF_RETURN_IF_ERROR(GetTaskName(dst_device, &task_prefix));
        const std::string src_device =
            absl::StrCat(task_prefix, "/device:CPU:0");
        const std::string& rendezvous_key =
            Rendezvous::CreateKey(src_device, /*src_incarnation=*/1, dst_device,
                                  rendezvous_key_base, FrameAndIter(0, 0));
        VLOG(2) << "Rendezvous Key for sending from host: " << rendezvous_key;

        RendezvousInterface::ParsedKey parsed_key;
        TF_RETURN_IF_ERROR(Rendezvous::ParseKey(rendezvous_key, &parsed_key));

        TF_ASSIGN_OR_RETURN(auto event, stream->parent()->CreateEvent());
        tsl::AsyncValueRef<std::unique_ptr<se::Event>> done_event =
            tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
                std::move(event));

        Rendezvous::Args args;
        // Rendezvous::Args owns the device context pointer.
        args.device_context = new XlaHostSendDeviceContext(
            stream, device_memory_base, shape, done_event);

        Tensor device_tensor;
        bool is_dead;
        TF_RETURN_IF_ERROR(ctx->rendezvous()->Recv(
            parsed_key, args, &device_tensor, /*is_dead=*/&is_dead));

        return std::move(done_event);
      };
}

absl::StatusOr<xla::ExecutionOutput> RunExecutable(
    const XlaPlatformInfo& platform_info,
    const XlaComputationLaunchContext& launch_context,
    std::vector<xla::ExecutionInput> execution_inputs,
    xla::ExecutableRunOptions run_options, xla::LocalExecutable* executable,
    OpKernelContext* ctx, se::DeviceMemoryAllocator* allocator) {
  VLOG(2) << "Executing Xla Computation.";
  Env* env = Env::Default();
  auto start_time = env->NowMicros();

  se::Stream* stream = GetStream(ctx);
  run_options.set_stream(GetStream(ctx));
  run_options.set_allocator(allocator);
  run_options.set_intra_op_thread_pool(&ctx->eigen_cpu_device());
  run_options.set_rng_seed(GetXLARandomSeed());

  absl::StatusOr<xla::ExecutionOutput> execution_output;
  bool run_synchronous =
      !stream || platform_info.platform_id() == se::host::kHostPlatformId;
  if (run_synchronous) {
    execution_output =
        executable->Run(std::move(execution_inputs), run_options);
  } else {
    execution_output =
        executable->RunAsync(std::move(execution_inputs), run_options);
  }

  auto elapsed = env->NowMicros() - start_time;
  VLOG(2) << "Elapsed time for Xla Executable Run: " << elapsed << "us";
  return execution_output;
}

absl::StatusOr<
    std::pair<std::vector<XlaCompiler::Argument>, ResourceVarsSnapshot>>
GetXlaCompilerArgsAndSnapshotVariables(
    absl::Span<const int> variable_indices,
    absl::Span<const int> must_be_constant_idxs,
    absl::Span<const Tensor* const> inputs, OpKernelContext* ctx) {
  std::pair<std::vector<XlaCompiler::Argument>, ResourceVarsSnapshot> result;

  std::vector<VariableInfo> variable_infos;
  TF_RETURN_IF_ERROR(
      GetVariableInfosFromInputs(ctx->resource_manager(), ctx->device(), inputs,
                                 variable_indices, &variable_infos));
  TF_RETURN_IF_ERROR(LockVariables(absl::MakeSpan(variable_infos)));

  TF_RETURN_IF_ERROR(SnapshotResourceVariables(ctx, variable_indices,
                                               variable_infos, &result.second));

  TF_ASSIGN_OR_RETURN(result.first,
                      XlaComputationLaunchContext::BuildXlaCompilerArguments(
                          must_be_constant_idxs, inputs, variable_infos,
                          static_cast<Device*>(ctx->device())));
  return result;
}

absl::Status CompileToLocalExecutable(
    OpKernelContext* ctx, const NameAttrList& function, bool has_ref_vars,
    const XlaPlatformInfo& platform_info,
    const std::vector<XlaCompiler::Argument>& args,
    DeviceCompileMode compile_mode, bool may_alias_resource_update,
    xla::LocalClient** client,
    const XlaCompiler::CompilationResult** compilation_result,
    xla::LocalExecutable** executable) {
  // We store information about the JIT-compiled XLA computation
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  if (!rm) {
    return absl::InternalError("No resource manager.");
  }

  TF_ASSIGN_OR_RETURN(DeviceType compilation_device_type,
                      GetCompilationDeviceType(platform_info.device_type()));

  XlaDeviceCompiler* xla_device_compiler;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<XlaDeviceCompiler>(
      rm->default_container(), "xla_device_compiler", &xla_device_compiler,
      [&](XlaDeviceCompiler** xla_device_compiler) {
        return BuildXlaDeviceCompiler(ctx->device(), ctx->function_library(),
                                      platform_info, compilation_device_type,
                                      xla_device_compiler);
      }));
  DeviceCompilationProfiler* profiler;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<DeviceCompilationProfiler>(
      rm->default_container(), "device_compilation_profiler", &profiler,
      [](DeviceCompilationProfiler** profiler) {
        *profiler = new DeviceCompilationProfiler();
        return absl::OkStatus();
      }));
  // Hold the reference to the XLA device compiler and profiler during
  // evaluation. (We could probably free them sooner because the ResourceMgr
  // will retain references, but this is more obviously correct.)
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);
  core::ScopedUnref profiler_ref(profiler);

  *client = static_cast<xla::LocalClient*>(xla_device_compiler->client());

  XlaCompiler::Options options = GenerateCompilerOptions(
      *xla_device_compiler, *ctx->function_library(), ctx->device(),
      GetStream(ctx), platform_info, has_ref_vars);

  XlaCompiler::CompileOptions compile_options =
      GenerateCompileOptions(has_ref_vars, may_alias_resource_update);

  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  if (flags->tf_xla_enable_dynamic_sizes) {
    // Rewriting the argument with expressions if they have dynamic
    // dimension, detecting dynamic dimension via either _dynamic_dim or the
    // inferred-output-shapes attr attached during encapsulation.
    std::vector<XlaCompiler::Argument> norm_args(args.begin(), args.end());
    int64_t filled_batch = 0;
    bool saw_dynamic_dim_value = false;
    // Only supporting one dynamic dimension. 
    bool has_multiple_dynamic_dim_values = false;
    int64_t dynamic_dim_value = 0;
    XlaBatchMatcher* xla_batch_matcher =
        xla_device_compiler->xla_batch_matcher();
    std::optional<xla::DExpr> dynamic_dim_expr;
    std::optional<xla::DExpr> shared_dynamic_subexpr;
    auto normalize_dynamic_expr =
        [&](xla::DExpr expr, absl::string_view context = "") {
          if (!expr || !expr->is_dynamic()) {
            return expr;
          }
          if (!shared_dynamic_subexpr.has_value()) {
            shared_dynamic_subexpr =
                expr.find_smallest_subexpression_covering_all_variables();
            LOG(INFO) << "Using shared dynamic subexpression "
                      << DExprToString(*shared_dynamic_subexpr)
                      << " for XLA dynamic input normalization. context="
                      << context;
          }
          xla::DExpr normalized_expr =
              expr.replace_subexpression(*shared_dynamic_subexpr,
                                         xla::DExpr::Var(1))
                  .simplify();
          LOG(INFO) << "Rewriting dynamic input expression "
                    << DExprToString(expr)
                    << " to shared-core form "
                    << DExprToString(normalized_expr)
                    << " before XLA compilation. context=" << context;
          return normalized_expr;
        };
    auto maybe_attach_shape_contents_from_attrs =
        [&](int arg_index, const auto& attr_map,
            const std::string& node_name) {
          auto& arg = norm_args[arg_index];
          if (arg.kind != XlaCompiler::Argument::kConstant) {
            return;
          }

          auto inferred_shape_it = attr_map.find("user_inferred_shape");
          auto inferred_contents_it =
              attr_map.find(kUserInferredValueContentsAttrName);
          bool has_dynamic = false;
          auto has_dynamic_it = attr_map.find("has_dynamic");
          if (has_dynamic_it != attr_map.end()) {
            has_dynamic = has_dynamic_it->second.b();
          }

          if (inferred_contents_it == attr_map.end() &&
              (!has_dynamic || inferred_shape_it == attr_map.end())) {
            return;
          }

          TensorShapeProto inferred_shape_proto;
          if (inferred_contents_it != attr_map.end()) {
            if (!inferred_shape_proto.ParseFromString(
                    inferred_contents_it->second.s())) {
              return;
            }
          } else {
            inferred_shape_proto = inferred_shape_it->second.shape();
          }

          TensorShape inferred_shape(inferred_shape_proto);
          if (!((TensorShapeUtils::IsVector(arg.constant_value.shape()) &&
                 arg.constant_value.NumElements() == inferred_shape.dims()) ||
                (TensorShapeUtils::IsScalar(arg.constant_value.shape()) &&
                 inferred_shape.dims() == 1))) {
            return;
          }

          arg.constant_value_expressions.clear();
          arg.constant_value_expressions.reserve(inferred_shape.dims());
          for (int64_t i = 0; i < inferred_shape.dims(); ++i) {
            xla::ExpressionProto expr;
            const xla::DExpr& dim_expr = inferred_shape.get_expression(i);
            if (dim_expr && dim_expr->is_dynamic()) {
              xla::DExpr normalized_expr = normalize_dynamic_expr(
                  dim_expr,
                  absl::StrCat("const_arg=", arg_index, " node=", node_name,
                               " dim=", i));
              normalized_expr->to_proto(&expr);
            } else if (arg.constant_value.dtype() == DT_INT32) {
              expr.set_constant_value(arg.constant_value.flat<int32>()(i));
            } else if (arg.constant_value.dtype() == DT_INT64) {
              expr.set_constant_value(arg.constant_value.flat<int64_t>()(i));
            } else {
              arg.constant_value_expressions.clear();
              return;
            }
            arg.constant_value_expressions.push_back(std::move(expr));
          }
        };
    auto record_dynamic_dim_value = [&](int64_t dim_size, xla::DExpr expr) {
      if (!saw_dynamic_dim_value) {
        saw_dynamic_dim_value = true;
        dynamic_dim_value = dim_size;
        dynamic_dim_expr = std::move(expr);
        return;
      }
      if (dynamic_dim_value != dim_size) {
        has_multiple_dynamic_dim_values = true;
      }
    };
    if (options.flib_def != nullptr) {
      const FunctionDef* fdef = options.flib_def->Find(function.name());
      if (fdef != nullptr) {
        for (const auto& kv : fdef->arg_attr()) {
          int arg_index = kv.first;
          const auto& attr_map = kv.second.attr();
          const std::string& node_name =
              fdef->signature().input_arg(arg_index).name();

          auto shape_derived_attr = attr_map.find(kXlaShapeDerivedAttrName);
          if (shape_derived_attr != attr_map.end()) {
            VLOG(1) << "XlaCompileOp retrieved shape-derived marker for arg "
                    << arg_index << " node=" << node_name;
          }
          maybe_attach_shape_contents_from_attrs(arg_index, attr_map, node_name);

          // Special case for _dynamic_dim...
          auto dyn_dim_attr = attr_map.find("_dynamic_dim");
          if (dyn_dim_attr != attr_map.end()) {
            TensorShape& shp =
                std::get<TensorShape>(norm_args[arg_index].shape);
            const AttrValue& v = dyn_dim_attr->second;
            int64_t idx = v.i();
            record_dynamic_dim_value(shp.dim_size(idx), xla::DExpr::Var(1));
            if (!filled_batch && xla_batch_matcher) {
              filled_batch =
                  xla_batch_matcher->get_xla_compile_batch(shp.dim_size(idx));
            }

            std::vector<xla::DExpr> dyn_exprs;
            for (int d : shp.dim_sizes()) {
              dyn_exprs.push_back(xla::DExpr::Const(d));
            }
            dyn_exprs[idx] = *dynamic_dim_expr;
            shp.set_expressions(std::move(dyn_exprs));
            continue;
          }
          auto it = attr_map.find(kXlaInferredOutputShapesAttrName);
          if (it == attr_map.end()) continue;

          const TensorShapeProto& proto = it->second.list().shape(0);
          const auto& exp = proto.expressions();
          TensorShape& shp = std::get<TensorShape>(norm_args[arg_index].shape);
          if (!filled_batch && xla_batch_matcher) {
            for (int idx = 0; idx < exp.size(); ++idx) {
              // Look for dynamic expression. If found then compute padding
              // value and exit loop.
              auto e = normalize_dynamic_expr(DimExprFromProto(exp[idx]));
              if (e->is_dynamic()) {
                LOG(INFO) << "Calling dynamic expression solve for compile "
                          << "argument " << arg_index << " dimension " << idx
                          << " expr=" << DExprToString(e)
                          << " target_size=" << shp.dim_size(idx);
                std::optional<int64_t> solved_value =
                    e->solve(shp.dim_size(idx));
                LOG(INFO) << "Dynamic expression solve for compile argument "
                          << arg_index << " dimension " << idx << " returned "
                          << (solved_value.has_value()
                                  ? std::to_string(*solved_value)
                                  : std::string("<none>"));
                int64_t var_value;
                if (!solved_value.has_value()) {
                  LOG(WARNING)
                      << "Failed to solve dynamic dimension for argument "
                      << arg_index << " dim " << idx << " with size "
                      << shp.dim_size(idx)
                      << "; falling back to original dimension size.";
                  var_value = shp.dim_size(idx);
                } else {
                  var_value = *solved_value;
                  VLOG(1) << "Solved dynamic dimension from "
                          << shp.dim_size(idx) << " to " << var_value;
                }
                record_dynamic_dim_value(var_value, e);
                filled_batch =
                    xla_batch_matcher->get_xla_compile_batch(var_value);
                break;
              }
            }
          }

          std::vector<xla::DExpr> dyn_exprs;
          for (int d : shp.dim_sizes()) {
            dyn_exprs.push_back(xla::DExpr::Const(d));
          }
          for (int j = 0; j < exp.size(); ++j) {
            auto e = DimExprFromProto(exp[j]);
            if (e->is_dynamic()) {
              e = normalize_dynamic_expr(
                  e, absl::StrCat("arg=", arg_index, " dim=", j,
                                  " input_shape"));
              dyn_exprs[j] = e;
            }
          }
          shp.set_expressions(std::move(dyn_exprs));
        }
      }
    }

    DynamicSolveFilterDecision solve_filter_decision =
        AnalyzeIgnoredDynamicArgumentOccurrences(norm_args);
    if (!solve_filter_decision.can_run) {
      if (compile_mode == DeviceCompileMode::kLazy) {
        return errors::Unimplemented(solve_filter_decision.diagnostic);
      }
      return errors::InvalidArgument(solve_filter_decision.diagnostic);
    }
    if (!solve_filter_decision.ignored_occurrences.empty()) {
      LOG(INFO) << solve_filter_decision.diagnostic;
      StripIgnoredDynamicArgumentOccurrences(
          solve_filter_decision.ignored_occurrences, &norm_args);
      saw_dynamic_dim_value = false;
      has_multiple_dynamic_dim_values = false;
      dynamic_dim_value = 0;
      dynamic_dim_expr.reset();
      filled_batch = 0;
      for (int arg_index = 0; arg_index < norm_args.size(); ++arg_index) {
        if (!absl::holds_alternative<TensorShape>(norm_args[arg_index].shape)) {
          continue;
        }
        TensorShape& shape = std::get<TensorShape>(norm_args[arg_index].shape);
        for (int dim = 0; dim < shape.get_expressions().size(); ++dim) {
          xla::DExpr expr = shape.get_expression(dim);
          if (!(expr && expr->is_dynamic())) {
            continue;
          }
          xla::DExpr simplified_expr = expr.simplify();
          std::optional<int64_t> solved_value =
              simplified_expr->solve(shape.dim_size(dim));
          if (!solved_value.has_value()) {
            continue;
          }
          record_dynamic_dim_value(*solved_value, simplified_expr);
          if (!filled_batch && xla_batch_matcher) {
            filled_batch =
                xla_batch_matcher->get_xla_compile_batch(*solved_value);
          }
        }
      }
    }

    struct SaveOldVar {
      int arg_index;
      int64_t dyn_dim;
      int64_t old_value;
    };
    std::vector<SaveOldVar> old_vars;
    auto maybe_rewrite_scalar_constant = [&](int arg_index) {
      if (!saw_dynamic_dim_value || has_multiple_dynamic_dim_values) {
        return;
      }

      auto& arg = norm_args[arg_index];
      if (arg.kind != XlaCompiler::Argument::kConstant) {
        return;
      }

      const bool is_scalar = TensorShapeUtils::IsScalar(arg.constant_value.shape());
      const bool is_vector = TensorShapeUtils::IsVector(arg.constant_value.shape()) &&
                             arg.constant_value.NumElements() > 0;
      if (!is_scalar && !is_vector) {
        return;
      }

      auto set_constant_contents = [&]<typename T>(int rewrite_index) {
        arg.constant_value_expressions.clear();
        const int64_t num_elements = arg.constant_value.NumElements();
        arg.constant_value_expressions.reserve(num_elements);
        for (int64_t i = 0; i < num_elements; ++i) {
          xla::ExpressionProto expr;
          if (i == rewrite_index) {
            dynamic_dim_expr->to_proto(&expr);
          } else {
            expr.set_constant_value(arg.constant_value.flat<T>()(i));
          }
          arg.constant_value_expressions.push_back(std::move(expr));
        }
      };

      if (arg.constant_value.dtype() == DT_INT32) {
        auto flat = arg.constant_value.flat<int32>();
        int rewrite_index = -1;
        // Heuristic: rewrite only scalar constants or shape-like int vectors.
        // In practice we expect at most one entry to match the observed
        // runtime batch size, so rewrite the first matching entry.
        for (int i = 0; i < arg.constant_value.NumElements(); ++i) {
          if (flat(i) == dynamic_dim_value) {
            rewrite_index = i;
            break;
          }
        }
        if (rewrite_index >= 0) {
          arg.constant_value = tensor::DeepCopy(arg.constant_value);
          auto mutable_flat = arg.constant_value.flat<int32>();
          VLOG(1) << "XlaCompileOp int32 constant arg " << arg_index
                  << " index " << rewrite_index
                  << " matches dynamic_dim_value=" << dynamic_dim_value;
          mutable_flat(rewrite_index) = filled_batch;
          set_constant_contents.template operator()<int32>(rewrite_index);
        }
      } else if (arg.constant_value.dtype() == DT_INT64) {
        auto flat = arg.constant_value.flat<int64_t>();
        int rewrite_index = -1;
        // Same heuristic for int64 scalar constants or shape-like vectors.
        for (int i = 0; i < arg.constant_value.NumElements(); ++i) {
          if (flat(i) == dynamic_dim_value) {
            rewrite_index = i;
            break;
          }
        }
        if (rewrite_index >= 0) {
          arg.constant_value = tensor::DeepCopy(arg.constant_value);
          auto mutable_flat = arg.constant_value.flat<int64_t>();
          VLOG(1) << "XlaCompileOp int64 constant arg " << arg_index
                  << " index " << rewrite_index
                  << " matches dynamic_dim_value=" << dynamic_dim_value;
          mutable_flat(rewrite_index) = filled_batch;
          set_constant_contents.template operator()<int64_t>(rewrite_index);
        }
      }
    };
    // We rewrite only dynamic dimensions to the padded compile batch and then
    // restore the original runtime sizes after compilation. Some scalar
    // constants are actually runtime batch sizes folded by earlier TF passes,
    // so rewrite only those that match the detected dynamic runtime value.
    // Scalar constants are deep-copied before rewrite so the change stays
    // local to norm_args and does not require restoration.
    if (filled_batch) {
      for (int i = 0; i < norm_args.size(); ++i) {
        TensorShape& shp = std::get<TensorShape>(norm_args[i].shape);
        for (int j = 0; j < shp.get_expressions().size(); ++j) {
          auto e = shp.get_expression(j);
          if (e && e->is_dynamic()) {
            int64_t old = shp.dim_size(j);
            old_vars.push_back({i, j, old});
            xla::DExpr padded_expr = xla::DExpr::Const(filled_batch);
            const std::set<int> ids = e->get_all_ids();
            if (ids.size() != 1) {
              return errors::InvalidArgument(
                  "Dynamic shape padding expected exactly one dynamic "
                  "variable for argument ",
                  i, ", dimension ", j, ", but found ", ids.size(),
                  " variables in expression ", DExprToString(e));
            }
            const int substitute_var_id = *ids.begin();
            LOG(INFO) << "Calling dynamic expression substitute for compile "
                      << "argument " << i << " dimension " << j
                      << " expr=" << DExprToString(e)
                      << " substitute Var(" << substitute_var_id
                      << ")=" << filled_batch;
            xla::DExpr subst_expr =
                e.substitute(substitute_var_id, padded_expr).simplify();
            LOG(INFO) << "Dynamic expression substitute for compile argument "
                      << i << " dimension " << j
                      << " returned " << DExprToString(subst_expr);
            if (!subst_expr->is_constant()) {
              return errors::InvalidArgument(
                  "Dynamic shape padding substitution did not produce an "
                  "integer constant for argument ",
                  i, ", dimension ", j, ": ", DExprToString(subst_expr));
            }
            int64_t new_dim = subst_expr->get_val();
            if (new_dim >= 0) {
              shp.set_dim(j, new_dim);
              // Necessary because set_dim removes the expression:
              shp.set_expression(j, e);
            }
          }
        }
        maybe_rewrite_scalar_constant(i);
      }
    }
    auto status = xla_device_compiler->CompileIfNeeded(
        options, function, norm_args, compile_options, compile_mode, profiler,
        compilation_result, executable);
    // Restore the original runtime dimensions after compilation.
    if (filled_batch) {
      for (const auto& old_var : old_vars) {
        TensorShape& shp =
            std::get<TensorShape>(norm_args[old_var.arg_index].shape);
        shp.set_dim(old_var.dyn_dim, old_var.old_value);
      }
    }
    return status;
  } else {
    return xla_device_compiler->CompileIfNeeded(
        options, function, args, compile_options, compile_mode, profiler,
        compilation_result, executable);
  }
}

absl::Status GetUpdatedVariables(
    const OpKernelContext* ctx, absl::Span<const Tensor* const> inputs,
    absl::Span<const int> variable_indices,
    const XlaCompiler::CompilationResult& compilation_result,
    std::vector<VariableInfo>* variable_infos) {
  std::set<int> variables_updated;
  for (const auto& resource_update : compilation_result.resource_updates) {
    if (resource_update.modified) {
      variables_updated.insert(resource_update.input_index);
    }
  }
  return GetVariableInfosFromInputs(ctx->resource_manager(), ctx->device(),
                                    inputs, variable_indices,
                                    &variables_updated, variable_infos);
}

// Get-or-create thread pool for a given collective.
static thread::ThreadPool* GetOrCreateThreadPoolForCollective(
    const XlaCompilationResult::CollectiveInfo& collective_info) {
  static absl::Mutex m(absl::kConstInit);
  static auto& thread_pool_cache ABSL_GUARDED_BY(m) =
      *new absl::node_hash_map<XlaCompilationResult::CollectiveInfo,
                               thread::ThreadPool>();
  absl::MutexLock l(&m);
  auto it = thread_pool_cache.find(collective_info);
  if (it == thread_pool_cache.end()) {
    // Create & cache thread pool.
    auto inserted_it = thread_pool_cache.emplace(
        std::piecewise_construct, std::forward_as_tuple(collective_info),
        std::forward_as_tuple(Env::Default(), "xla_collective_thread_pool",
                              collective_info.group_size));
    return &inserted_it.first->second;
  }
  return &it->second;
}

void RunInThreadPoolIfCollectivesPresent(
    const XlaCompiler::CompilationResult& compilation_result,
    std::function<void()> execution_fn) {
  // If we are using collectives, we need to run in a separate threadpool.
  if (compilation_result.collective_info.has_value()) {
    GetOrCreateThreadPoolForCollective(*compilation_result.collective_info)
        ->Schedule(execution_fn);
  } else {
    // Otherwise, just run normally: we merely "pretend" to be asynchronous.
    execution_fn();
  }
}

}  // namespace

XlaLocalLaunchBase::XlaLocalLaunchBase(OpKernelConstruction* ctx,
                                       const std::vector<int>& constants,
                                       const std::vector<int>& resources,
                                       const NameAttrList& function,
                                       bool has_ref_vars)
    : AsyncOpKernel(ctx),
      constants_(constants),
      resources_(resources),
      function_(function),
      platform_info_(XlaPlatformInfoFromDevice(ctx->device())),
      has_ref_vars_(has_ref_vars) {}

void XlaLocalLaunchBase::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  VLOG(1) << "XlaLocalLaunchOpBase::Compute "
          << Canonicalize(function_.name(), AttrSlice(&function_.attr()));
  xla_launch_counter->GetCell(platform_info_.device_type().type_string())
      ->IncrementBy(1);

  std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  std::vector<XlaCompiler::Argument> xla_compiler_args;
  const XlaCompiler::CompilationResult* compilation_result;

  xla::LocalClient* client;          // Not owned.
  xla::LocalExecutable* executable;  // Not owned.

  xla::PjRtClient* pjrt_client;                // Not owned.
  xla::PjRtLoadedExecutable* pjrt_executable;  // Not owned.

  // Note that here we assume the shape of the variables don't change between
  // compilation and execution. The locks on the variables are released before
  // compilation so that we can achieve parallel compilation of different batch
  // sizes during warm-up.
  {
    // Creating a scope so that the locks on the variables are released when
    // variable_infos goes out of scope.
    std::vector<VariableInfo> variable_infos;
    std::set<int> variables_updated;
    // Here we only need to reader-lock the variables, so we pass an empty
    // variables_updated set here.
    absl::Status status = GetVariableInfosFromInputs(
        ctx->resource_manager(), ctx->device(), inputs, resources_,
        &variables_updated, &variable_infos);
    OP_REQUIRES_OK_ASYNC(ctx, status, done);
    status = LockVariables(absl::MakeSpan(variable_infos));
    OP_REQUIRES_OK_ASYNC(ctx, status, done);
    auto status_or_xla_compiler_args =
        XlaComputationLaunchContext::BuildXlaCompilerArguments(
            constants_, inputs, variable_infos,
            static_cast<Device*>(ctx->device()));
    OP_REQUIRES_OK_ASYNC(ctx, status_or_xla_compiler_args.status(), done);
    xla_compiler_args = std::move(status_or_xla_compiler_args.value());
  }

  bool use_pjrt = GetXlaOpsCommonFlags()
                      ->tf_xla_use_device_api.IsEnabledInXlaLaunchForDevice(
                          platform_info_.device_type());
  if (use_pjrt) {
    VLOG(2) << "Compiling using PJRT";
    absl::Status status = CompileToPjRtLoadedExecutable(
        *ctx, platform_info_, function_, xla_compiler_args,
        DeviceCompileMode::kStrict, has_ref_vars_,
        /*may_alias_resource_update=*/true, &compilation_result, &pjrt_client,
        &pjrt_executable);
    OP_REQUIRES_OK_ASYNC(ctx, status, done);

    VLOG(2) << "Compiled using PJRT: " << status;
    VLOG(2) << "pjrt_executable != nullptr: " << (pjrt_executable != nullptr);
    VLOG(2) << "compilation_result != nullptr: "
            << (compilation_result != nullptr);
    VLOG(2) << "Executing using PJRT.";

    auto run_pjrt_cluster = [ctx, pjrt_client, pjrt_executable,
                             compilation_result, done, inputs,
                             resources = resources_]() {
      // Separate scope so that VariableInfo locks are released before done() is
      // called.
      {
        std::vector<VariableInfo> variable_infos;
        OP_REQUIRES_OK_ASYNC(
            ctx,
            GetUpdatedVariables(ctx, inputs, resources, *compilation_result,
                                &variable_infos),
            done);
        OP_REQUIRES_OK_ASYNC(ctx, LockVariables(absl::MakeSpan(variable_infos)),
                             done);
        OP_REQUIRES_OK_ASYNC(
            ctx,
            RunPjRtExecutable(inputs, variable_infos, *compilation_result,
                              pjrt_client, pjrt_executable, ctx),
            done);
      }
      VLOG(2) << "Done executing with PJRT.";
      done();
    };

    RunInThreadPoolIfCollectivesPresent(*compilation_result, run_pjrt_cluster);
    return;
  }

  absl::Status status = CompileToLocalExecutable(
      ctx, function_, /*has_ref_vars=*/has_ref_vars_, platform_info_,
      xla_compiler_args, DeviceCompileMode::kStrict,
      /*may_alias_resource_update=*/true, &client, &compilation_result,
      &executable);
  OP_REQUIRES_OK_ASYNC(ctx, status, done);

  // Continuation of the execution, may be run in a different thread.
  auto run_xla_cluster = [ctx, client, executable, compilation_result, done,
                          inputs, resources = resources_]() {
    // Separate scope so that VariableInfo locks are released before done is
    // called.
    {
      auto platform_info = XlaPlatformInfoFromDevice(ctx->device());
      std::vector<VariableInfo> variable_infos;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          GetUpdatedVariables(ctx, inputs, resources, *compilation_result,
                              &variable_infos),
          done);
      OP_REQUIRES_OK_ASYNC(ctx, LockVariables(absl::MakeSpan(variable_infos)),
                           done);
      std::map<int, const Tensor*> resource_var_ptrs;
      for (int i = 0; i < resources.size(); i++) {
        resource_var_ptrs[resources[i]] = variable_infos[i].var()->tensor();
      }

      std::shared_ptr<se::DeviceMemoryAllocator> allocator =
          GetAllocator(ctx->device(), GetStream(ctx), platform_info);
      XlaComputationLaunchContext launch_context =
          GetLaunchContext(platform_info, ctx, client, allocator.get());

      const xla::HloInputOutputAliasConfig& input_output_alias =
          executable->executable()->module().input_output_alias_config();
      absl::StatusOr<std::vector<xla::ExecutionInput>> execution_inputs =
          launch_context.PopulateInputs(
              ctx, compilation_result, resource_var_ptrs,
              /*missing_ctx_input_prefix=*/0, input_output_alias);
      OP_REQUIRES_OK_ASYNC(ctx, execution_inputs.status(), done);

      xla::gpu::GpuExecutableRunOptions gpu_options;
      xla::DeviceAssignment device_assignment;
      xla::ExecutableRunOptions run_options;
      if (compilation_result->collective_info.has_value()) {
        OP_REQUIRES_OK_ASYNC(ctx,
                             ResolveDeviceAssignment(
                                 ctx, *compilation_result->collective_info,
                                 run_options, device_assignment, gpu_options),
                             done);
      }

      // Hardcode run id to always be zero: TF distributed strategy
      // differentiates between subsequent runs using dependency edges. This
      // is safe, as only TF dist-strat can produce distributed ops, and we
      // can rely on TF dist-strat invariants.
      xla::RunId run_id(0);
      run_options.set_run_id(run_id);

      absl::StatusOr<xla::ExecutionOutput> execution_output = RunExecutable(
          platform_info, launch_context, std::move(*execution_inputs),
          run_options, executable, ctx, allocator.get());
      OP_REQUIRES_ASYNC(ctx, execution_output.ok(), execution_output.status(),
                        done);

      OP_REQUIRES_OK_ASYNC(
          ctx,
          launch_context.PopulateOutputs(
              ctx, compilation_result, execution_output->ConsumeResult(),
              /*missing_ctx_input_prefix=*/0, absl::MakeSpan(variable_infos),
              input_output_alias, resource_var_ptrs),
          done);
      VLOG(1) << "Done";
    }
    done();
  };

  RunInThreadPoolIfCollectivesPresent(*compilation_result, run_xla_cluster);
}

namespace {
// Helper static functions to construct parameters for
// XlaLocalLaunchBase constructor from OpKernelConstruction.
std::vector<int> ConstantsVector(OpKernelConstruction* ctx) {
  DataTypeVector constant_types;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Tconstants", &constant_types));
  std::vector<int> constants(constant_types.size());
  std::iota(constants.begin(), constants.end(), 0);
  return constants;
}

std::vector<int> ResourcesVector(OpKernelConstruction* ctx) {
  DataTypeVector constant_types;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Tconstants", &constant_types));

  DataTypeVector arg_types;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Targs", &arg_types));

  int num_resources;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(),
                        ctx->GetAttr("Nresources", &num_resources));

  std::vector<int> resources(num_resources);
  std::iota(resources.begin(), resources.end(),
            constant_types.size() + arg_types.size());
  return resources;
}

NameAttrList FunctionAttr(OpKernelConstruction* ctx) {
  const NameAttrList* func;
  OP_REQUIRES_OK_RETURN(ctx, NameAttrList(), ctx->GetAttr("function", &func));
  return *func;
}

std::vector<int> VectorAttr(OpKernelConstruction* ctx,
                            absl::string_view attr_name) {
  std::vector<int> vec;
  OP_REQUIRES_OK_RETURN(ctx, std::vector<int>(), ctx->GetAttr(attr_name, &vec));
  return vec;
}

bool MustCompileAttr(OpKernelConstruction* ctx) {
  bool must_compile;
  OP_REQUIRES_OK_RETURN(ctx, false,
                        ctx->GetAttr("must_compile", &must_compile));
  return must_compile;
}

bool HasRefVars(OpKernelConstruction* ctx) {
  bool has_ref_vars;
  OP_REQUIRES_OK_RETURN(ctx, false,
                        ctx->GetAttr(kXlaHasReferenceVarsAttr, &has_ref_vars));
  return has_ref_vars;
}

class XlaLaunchV2Op : public XlaLocalLaunchBase {
 public:
  explicit XlaLaunchV2Op(OpKernelConstruction* ctx)
      : XlaLocalLaunchBase(ctx, VectorAttr(ctx, "constants"),
                           VectorAttr(ctx, "resources"), FunctionAttr(ctx),
                           /*has_ref_vars=*/true) {}
};

}  // namespace

XlaLocalLaunchOp::XlaLocalLaunchOp(OpKernelConstruction* ctx)
    : XlaLocalLaunchBase(ctx, ConstantsVector(ctx), ResourcesVector(ctx),
                         FunctionAttr(ctx), /*has_ref_vars=*/true) {}

XlaLocalLaunchOp::~XlaLocalLaunchOp() {
  VLOG(1) << "XlaLocalLaunchOp destroyed";
}

XlaCompileOp::XlaCompileOp(OpKernelConstruction* ctx)
    : OpKernel(ctx),
      constants_(ConstantsVector(ctx)),
      resources_(ResourcesVector(ctx)),
      function_(FunctionAttr(ctx)),
      platform_info_(XlaPlatformInfoFromDevice(ctx->device())),
      must_compile_(MustCompileAttr(ctx)),
      has_ref_vars_(HasRefVars(ctx)) {}

void XlaCompileOp::Compute(OpKernelContext* ctx) {
  VLOG(3) << "XlaCompileOp " << def().name()
          << (must_compile_ ? "(must-compile)" : "");
  const XlaCompiler::CompilationResult* kernel = nullptr;
  xla::LocalClient* client = nullptr;
  xla::LocalExecutable* executable = nullptr;
  xla::PjRtClient* pjrt_client = nullptr;
  xla::PjRtLoadedExecutable* pjrt_executable = nullptr;
  ResourceVarsSnapshot variables_snapshot;

  std::vector<const Tensor*> inputs = InputsFromContext(ctx);
  bool cannot_compile_cluster;
  {
    mutex_lock guard(cannot_compile_cluster_mu_);
    cannot_compile_cluster = cannot_compile_cluster_;
  }
  DeviceCompileMode compile_mode = [&] {
    if (must_compile_) {
      return DeviceCompileMode::kStrict;
    }
    return GetXlaOpsCommonFlags()->tf_xla_async_compilation
               ? DeviceCompileMode::kAsync
               : DeviceCompileMode::kLazy;
  }();

  bool use_pjrt =
      GetXlaOpsCommonFlags()
          ->tf_xla_use_device_api.IsEnabledInXlaCompileAndRunForDevice(
              platform_info_.device_type());

  if (GetXlaOpsCommonFlags()->tf_xla_always_defer_compilation ||
      cannot_compile_cluster) {
    executable = nullptr;
  } else {
    auto args_and_variables_snapshot = GetXlaCompilerArgsAndSnapshotVariables(
        resources_, constants_, inputs, ctx);
    OP_REQUIRES_OK(ctx, args_and_variables_snapshot.status());
    const std::vector<XlaCompiler::Argument>& args =
        args_and_variables_snapshot->first;
    variables_snapshot = std::move(args_and_variables_snapshot->second);

    // Do not alias resource updates as locking variables in XlaCompile and
    // unlocking them in XlaRun may lead to deadlocks.
    absl::Status status;
    if (use_pjrt) {
      VLOG(2) << "Using PJRT for compilation. Function name: "
              << function_.name();
      status = CompileToPjRtLoadedExecutable(
          *ctx, platform_info_, function_, args, compile_mode, has_ref_vars_,
          /*may_alias_resource_update=*/false, &kernel, &pjrt_client,
          &pjrt_executable);
    } else {
      status = CompileToLocalExecutable(
          ctx, function_, has_ref_vars_, platform_info_, args, compile_mode,
          /*may_alias_resource_update=*/false, &client, &kernel, &executable);
    }

    if (compile_mode != DeviceCompileMode::kLazy ||
        status.code() != error::UNIMPLEMENTED) {
      if ((status != OkStatus()) &&
          (status.code() != error::UNIMPLEMENTED) &&
        (compile_mode == DeviceCompileMode::kLazy)) {
        // We set the error to error::UNIMPLEMENTED so it falls in the
        // conditions of the if to fall back to TensorFlow function call
        status = tensorflow::errors::Unimplemented(status.ToString());
      } else {
        OP_REQUIRES_OK(ctx, status);
      }
    }

    if (status.code() == error::UNIMPLEMENTED) {
      LOG(WARNING) << "[HUAWEI] Compilation of the cluster failed with:";
      LOG(WARNING) << "[HUAWEI] " << status;
      LOG(WARNING) << "[HUAWEI] Falling back to TF function call.\n";

      BroadcastOptimizationRemark(
          XlaOptimizationRemark::UNIMPLEMENTED_OPERATION, status.ToString())
          .IgnoreError();
      executable = nullptr;
      pjrt_executable = nullptr;
      mutex_lock guard(cannot_compile_cluster_mu_);
      // TODO: decide if we want to set this flag to true, as we may want to
      // allow the cluster to try to compile again later in time.
      cannot_compile_cluster_ = true;
    }
  }

  if ((executable || pjrt_executable) && kernel != nullptr &&
      GetMarkForCompilationPassFlags()->tf_xla_enable_dynamic_sizes) {
    DynamicBatchResolutionResult resolution =
        ResolveDynamicBatchSizeFromRuntimeInputs(ctx, *kernel, constants_.size(),
                                                 /*log_solves=*/true,
                                                 function_.name(), def());
    if (!resolution.can_run) {
      const std::string error_message = absl::StrCat(
          "Rejecting XLA cluster at compile time because dynamic expressions "
          "cannot be solved consistently for this request. ",
          resolution.diagnostic);
      if (must_compile_) {
        OP_REQUIRES(ctx, false, errors::InvalidArgument(error_message));
      }
      LOG(WARNING) << error_message;
      executable = nullptr;
      pjrt_executable = nullptr;
      kernel = nullptr;
    }
  }

  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_gpu_compatible(true);
  host_alloc_attrs.set_on_host(true);
  Allocator* cpu_allocator = ctx->device()->GetAllocator(host_alloc_attrs);

  // Async compilation returns nullptr executable without an error.
  if (!executable && !pjrt_executable) {
    DCHECK(!must_compile_);
    Tensor compilation_key(cpu_allocator, DT_STRING, TensorShape({}));
    Tensor compilation_successful(cpu_allocator, DT_BOOL, TensorShape({}));
    compilation_successful.scalar<bool>()() = false;
    ctx->set_output(0, compilation_key);
    ctx->set_output(1, compilation_successful);
    return;
  }

  // Each execution of an XlaCompile op creates a new ExecutableClosure, even
  // if it didn't have to compile the cluster because of a compilation-cache
  // hit.  This is because we at least need new snapshots of the resource
  // variables.
  Tensor compilation_key(cpu_allocator, DT_STRING, TensorShape({}));
  if (use_pjrt) {
    PjRtExecutableClosureStore::KeyT key =
        PjRtExecutableClosureStore::Global()->Produce(PjRtExecutableClosure(
            pjrt_client, pjrt_executable, kernel, std::move(variables_snapshot),
            constants_.size()));
    compilation_key.flat<tstring>()(0) = key;
    VLOG(2) << "Compiled with PJRT. compilation_key: " << key;
  } else {
    XlaExecutableClosureStore::KeyT key =
        XlaExecutableClosureStore::Global()->Produce(XlaExecutableClosure(
            client, executable, kernel, std::move(variables_snapshot),
            constants_.size()));
    compilation_key.flat<tstring>()(0) = key;
    VLOG(2) << "Compiled with XLA. compilation_key: " << key;
  }

  Tensor compilation_successful(cpu_allocator, DT_BOOL, TensorShape({}));
  compilation_successful.flat<bool>()(0) = true;

  ctx->set_output(0, compilation_key);
  ctx->set_output(1, compilation_successful);
}

XlaRunOp::XlaRunOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), platform_info_(XlaPlatformInfoFromDevice(ctx->device())) {}

void XlaRunOp::Compute(OpKernelContext* ctx) {
  VLOG(3) << "XlaRunOp " << def().name();
  Tensor key_tensor = ctx->input(ctx->num_inputs() - 1);
  bool use_pjrt =
      GetXlaOpsCommonFlags()
          ->tf_xla_use_device_api.IsEnabledInXlaCompileAndRunForDevice(
              platform_info_.device_type());

  if (use_pjrt) {
    const PjRtExecutableClosureStore::KeyT& key = key_tensor.flat<tstring>()(0);
    PjRtExecutableClosure closure =
        PjRtExecutableClosureStore::Global()->Consume(key);
    const std::string cluster_name =
        closure.compilation_result() != nullptr &&
                closure.compilation_result()->computation != nullptr
            ? closure.compilation_result()->computation->name()
            : std::string("<unknown>");
    LOG(INFO) << "Entering XLA cluster: cluster=" << cluster_name
              << " op=" << def().name() << " key=" << key
              << " mode=pjrt step_id=" << ctx->step_id();

    // Fetch inputs from the OpKernelContext. Inputs are the same as the ones
    // for XlaCompile, except that the must-be-constant inputs that appear in
    // the beginning are stripped off and the closure key is appended as the
    // last input. So the inputs look like: input tensors, resource variables,
    // closure key tensor.
    std::vector<const Tensor*> inputs = InputsFromContext(ctx);
    absl::flat_hash_map<int, const Tensor*> variable_snapshots;
    for (const auto& [variable_index, variable_tensor] :
         closure.resource_var_snapshots()) {
      variable_snapshots.emplace(variable_index, variable_tensor.has_value()
                                                     ? &variable_tensor.value()
                                                     : nullptr);
    }

    {
      absl::StatusOr<std::vector<VariableInfo>> updated_variables =
          GatherVariableInfo(ctx, *closure.compilation_result(),
                             closure.num_constant_args());
      OP_REQUIRES_OK(ctx, updated_variables.status());
      OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(*updated_variables)));
      OP_REQUIRES_OK(
          ctx, RunPjRtExecutable(closure.num_constant_args(), inputs,
                                 variable_snapshots, *updated_variables,
                                 *closure.compilation_result(),
                                 closure.client(), closure.executable(), ctx));
    }

    OP_REQUIRES_OK(ctx, absl::OkStatus());
    return;
  }

  const XlaExecutableClosureStore::KeyT& key = key_tensor.flat<tstring>()(0);

  XlaExecutableClosure closure =
      XlaExecutableClosureStore::Global()->Consume(key);
  const std::string cluster_name =
      closure.compilation_result() != nullptr &&
              closure.compilation_result()->computation != nullptr
          ? closure.compilation_result()->computation->name()
          : std::string("<unknown>");
  LOG(INFO) << "Entering XLA cluster: cluster=" << cluster_name
            << " op=" << def().name() << " key=" << key
            << " mode=local step_id=" << ctx->step_id();
  std::shared_ptr<se::DeviceMemoryAllocator> allocator =
      GetAllocator(ctx->device(), GetStream(ctx), platform_info_);
  XlaComputationLaunchContext launch_context =
      GetLaunchContext(platform_info_, ctx, closure.client(), allocator.get());

  // We're missing the must-be-constant inputs, tell `PopulateInputs`
  // about this.  We don't actually need these inputs because they've
  // already been baked into the compiled kernel.
  const xla::HloInputOutputAliasConfig& input_output_alias =
      closure.executable()->executable()->module().input_output_alias_config();
  absl::StatusOr<std::vector<xla::ExecutionInput>> execution_inputs;
  std::map<int, const Tensor*> snapshot_ptrs;
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] {
          return absl::StrCat(
              "Populate Inputs (",
              closure.compilation_result()->xla_input_shapes.size(), ")");
        },
        tsl::profiler::TraceMeLevel::kInfo);

    for (const auto& [variable_index, variable_tensor] :
         closure.resource_var_snapshots()) {
      snapshot_ptrs.emplace(variable_index, variable_tensor.has_value()
                                                ? &variable_tensor.value()
                                                : nullptr);
    }
    execution_inputs = launch_context.PopulateInputs(
        ctx, closure.compilation_result(), snapshot_ptrs,
        /*missing_ctx_input_prefix=*/closure.num_constant_args(),
        input_output_alias);
    OP_REQUIRES_OK(ctx, execution_inputs.status());
  }

  xla::ExecutableRunOptions run_options;

  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  if (flags->tf_xla_enable_dynamic_sizes) {
    DynamicBatchResolutionResult batch_resolution =
        ResolveDynamicBatchSizeFromRuntimeInputs(
            ctx, *closure.compilation_result(), closure.num_constant_args(),
            /*log_solves=*/true, cluster_name, def());
    bool is_set = false;
    if (!batch_resolution.can_run) {
      LOG(ERROR) << batch_resolution.diagnostic;
      ctx->CtxFailure(errors::InvalidArgument(batch_resolution.diagnostic));
      return;
    }
    if (batch_resolution.has_batch_size) {
      LOG(INFO) << "Setting run_options.batch_size "
                << batch_resolution.diagnostic;
      run_options.set_batch_size(batch_resolution.batch_size);
      is_set = true;
    } else {
      LOG(INFO) << "Not setting run_options.batch_size because "
                << batch_resolution.diagnostic;
    }
    if (!is_set) {
      LOG(WARNING) << "Entering XLA cluster without run_options.batch_size "
                   << "being set. op=" << def().name() << " closure_key="
                   << key << " step_id=" << ctx->step_id()
                   << " current_run_options_batch_size="
                   << run_options.batch_size()
                   << " batch_size_resource_not_found="
                   << batch_resolution.batch_size_resource_not_found;
    }
  }

  // Host callbacks used for HLO send/recv.
  xla::SendDeviceMemoryFunction send_function =
      GetSendDeviceMemoryFunction(ctx, key);
  run_options.set_send_device_memory_function(&send_function);
  xla::RecvDeviceMemoryFunction recv_function =
      GetRecvDeviceMemoryFunction(ctx, key);
  run_options.set_recv_device_memory_function(&recv_function);

  absl::StatusOr<xla::ExecutionOutput> execution_output = RunExecutable(
      platform_info_, launch_context, std::move(*execution_inputs), run_options,
      closure.executable(), ctx, allocator.get());
  OP_REQUIRES(ctx, execution_output.ok(), execution_output.status());

  tsl::profiler::TraceMe hlo_module_activity(
      [&] {
        return absl::StrCat("Populate Outputs (", ctx->num_outputs(), ")");
      },
      tsl::profiler::TraceMeLevel::kInfo);

  absl::StatusOr<std::vector<VariableInfo>> variable_infos = GatherVariableInfo(
      ctx, *closure.compilation_result(), closure.num_constant_args());
  OP_REQUIRES_OK(ctx, variable_infos.status());
  OP_REQUIRES_OK(ctx, LockVariables(absl::MakeSpan(*variable_infos)));
  OP_REQUIRES_OK(
      ctx,
      launch_context.PopulateOutputs(
          ctx, closure.compilation_result(), execution_output->ConsumeResult(),
          /*missing_ctx_input_prefix=*/closure.num_constant_args(),
          absl::MakeSpan(*variable_infos), input_output_alias, snapshot_ptrs,
          &run_options));
}

XlaMergeOp::XlaMergeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

void XlaMergeOp::Compute(OpKernelContext* ctx) {
  VLOG(3) << "XlaMergeOp " << def().name();
  int i = 0;
  if (ctx->has_input(i) || ctx->has_input(++i)) {
    ctx->set_output(0, ctx->input(i));
  }
}

REGISTER_KERNEL_BUILDER(Name("XlaLaunch").Device(DEVICE_CPU), XlaLocalLaunchOp);

REGISTER_KERNEL_BUILDER(Name("XlaLaunchV2").Device(DEVICE_CPU), XlaLaunchV2Op);

REGISTER_KERNEL_BUILDER(Name("XlaLaunch")
                            .Device(DEVICE_GPU)
                            .HostMemory("constants")
                            .HostMemory("resources"),
                        XlaLocalLaunchOp);

REGISTER_KERNEL_BUILDER(Name("_XlaCompile").Device(DEVICE_CPU), XlaCompileOp);
REGISTER_KERNEL_BUILDER(Name("_XlaCompile")
                            .Device(DEVICE_GPU)
                            .HostMemory("constants")
                            .HostMemory("key")
                            .HostMemory("compilation_successful")
                            .HostMemory("resources"),
                        XlaCompileOp);

REGISTER_KERNEL_BUILDER(Name("_XlaCompile")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("constants")
                            .HostMemory("key")
                            .HostMemory("compilation_successful")
                            .HostMemory("resources"),
                        XlaCompileOp);

REGISTER_KERNEL_BUILDER(Name("_XlaRun").Device(DEVICE_CPU), XlaRunOp);
REGISTER_KERNEL_BUILDER(Name("_XlaRun").Device(DEVICE_GPU).HostMemory("key"),
                        XlaRunOp);
REGISTER_KERNEL_BUILDER(
    Name("_XlaRun").Device(DEVICE_DEFAULT).HostMemory("key"), XlaRunOp);

REGISTER_KERNEL_BUILDER(Name("_XlaMerge").Device(DEVICE_CPU), XlaMergeOp);
REGISTER_KERNEL_BUILDER(Name("_XlaMerge").Device(DEVICE_GPU), XlaMergeOp);
REGISTER_KERNEL_BUILDER(Name("_XlaMerge").Device(DEVICE_DEFAULT), XlaMergeOp);

}  // namespace tensorflow
