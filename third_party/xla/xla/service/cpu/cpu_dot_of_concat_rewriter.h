/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_CPU_DOT_OF_CONCAT_REWRITER_H_
#define XLA_SERVICE_CPU_CPU_DOT_OF_CONCAT_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::cpu {

// Rewrites narrow CPU dot patterns of the form:
//   dot(concatenate(lhs_0, lhs_1, ...), rhs)
// into:
//   dot(lhs_0, slice(rhs)) + dot(lhs_1, slice(rhs)) + ...
//
// This is an experimental CPU-only pass meant to avoid materializing a large
// concatenated LHS when it immediately fans out into dot operations.
class CpuDotOfConcatRewriter final : public HloModulePass {
 public:
  absl::string_view name() const final { return "cpu-dot-of-concat-rewriter"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) final;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_DOT_OF_CONCAT_REWRITER_H_
