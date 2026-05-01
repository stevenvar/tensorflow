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

#include "xla/service/cpu/cpu_dot_of_concat_rewriter.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"

namespace xla::cpu {
namespace {

bool IsSupportedRhs(const HloInstruction* rhs) {
  return rhs->opcode() == HloOpcode::kConstant ||
         rhs->opcode() == HloOpcode::kParameter;
}

absl::StatusOr<bool> TryRewriteDotOfConcat(HloInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      Cast<HloDotInstruction>(dot)->sparse_operands() ||
      dot->shape().dimensions().size() != 2) {
    return false;
  }

  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);
  const int64_t lhs_contracting_dim = dnums.lhs_contracting_dimensions(0);
  const int64_t rhs_contracting_dim = dnums.rhs_contracting_dimensions(0);

  if (lhs->opcode() != HloOpcode::kConcatenate ||
      lhs->concatenate_dimension() != lhs_contracting_dim ||
      lhs->operand_count() < 2 || !IsSupportedRhs(rhs) ||
      lhs->shape().dimensions().size() != 2 ||
      rhs->shape().dimensions().size() != 2) {
    return false;
  }
  const std::string dot_name(dot->name());

  DotDimensionNumbers new_dot_dnums;
  new_dot_dnums.add_lhs_contracting_dimensions(lhs_contracting_dim);
  new_dot_dnums.add_rhs_contracting_dimensions(rhs_contracting_dim);

  HloInstruction* add_result = nullptr;
  int64_t rhs_contracting_dim_offset = 0;
  const int64_t rhs_non_contracting_size =
      rhs->shape().dimensions(1 - rhs_contracting_dim);

  for (HloInstruction* concat_operand : lhs->operands()) {
    if (concat_operand->shape().dimensions().size() != 2) {
      return false;
    }

    const int64_t sub_k =
        concat_operand->shape().dimensions(lhs_contracting_dim);
    Shape rhs_slice_shape(rhs->shape());
    rhs_slice_shape.set_dimensions(rhs_contracting_dim, sub_k);

    std::array<int64_t, 2> start_indices;
    start_indices[rhs_contracting_dim] = rhs_contracting_dim_offset;
    start_indices[1 - rhs_contracting_dim] = 0;

    std::array<int64_t, 2> limit_indices;
    limit_indices[rhs_contracting_dim] = rhs_contracting_dim_offset + sub_k;
    limit_indices[1 - rhs_contracting_dim] = rhs_non_contracting_size;

    HloInstruction* rhs_slice = dot->AddInstruction(HloInstruction::CreateSlice(
        rhs_slice_shape, rhs, start_indices, limit_indices, {1, 1}));

    auto* new_dot = dot->AddInstruction(HloInstruction::CreateDot(
        dot->shape(), concat_operand, rhs_slice, new_dot_dnums,
        dot->precision_config()));
    dot->SetupDerivedInstruction(new_dot);

    add_result = add_result == nullptr
                     ? new_dot
                     : dot->AddInstruction(HloInstruction::CreateBinary(
                           dot->shape(), HloOpcode::kAdd, add_result, new_dot));

    rhs_contracting_dim_offset += sub_k;
  }

  TF_RETURN_IF_ERROR(dot->parent()->ReplaceInstruction(dot, add_result));
  LOG(INFO) << "CpuDotOfConcatRewriter rewrote dot: " << dot_name;
  return true;
}

}  // namespace

absl::StatusOr<bool> CpuDotOfConcatRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> dots;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kDot) {
        dots.push_back(instruction);
      }
    }
  }

  bool changed = false;
  for (HloInstruction* dot : dots) {
    TF_ASSIGN_OR_RETURN(bool rewrote, TryRewriteDotOfConcat(dot));
    changed |= rewrote;
  }
  return changed;
}

}  // namespace xla::cpu
