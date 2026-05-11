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

#include "xla/service/cpu/cpu_split_concat_dot.h"

#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

absl::StatusOr<bool> CpuSplitConcatDot::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kDot) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(bool rewritten, TryRewriteDot(instruction));
      changed |= rewritten;
    }
  }
  return changed;
}

absl::StatusOr<bool> CpuSplitConcatDot::TryRewriteDot(HloInstruction* dot) {
  const DotDimensionNumbers& dim_numbers = dot->dot_dimension_numbers();
  if (dim_numbers.lhs_contracting_dimensions_size() != 1 ||
      dim_numbers.rhs_contracting_dimensions_size() != 1 ||
      dim_numbers.lhs_batch_dimensions_size() != 0 ||
      dim_numbers.rhs_batch_dimensions_size() != 0) {
    return false;
  }

  HloInstruction* concat = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);
  if (concat->opcode() != HloOpcode::kConcatenate) {
    return false;
  }

  const int64_t lhs_contracting_dim =
      dim_numbers.lhs_contracting_dimensions(0);
  const int64_t rhs_contracting_dim =
      dim_numbers.rhs_contracting_dimensions(0);
  if (concat->concatenate_dimension() != lhs_contracting_dim ||
      concat->operand_count() < 2 ||
      concat->shape().dimensions().size() != 2 ||
      rhs->shape().dimensions().size() != 2 || !dot->shape().IsArray()) {
    return false;
  }
  if (concat->shape().dimensions(lhs_contracting_dim) !=
      rhs->shape().dimensions(rhs_contracting_dim)) {
    return false;
  }
  for (HloInstruction* concat_operand : concat->operands()) {
    if (concat_operand->shape().dimensions().size() !=
            concat->shape().dimensions().size() ||
        concat_operand->shape().element_type() !=
            concat->shape().element_type()) {
      return false;
    }
  }

  HloComputation* computation = dot->parent();
  std::vector<HloInstruction*> partial_dots;
  int64_t rhs_offset = 0;
  for (HloInstruction* concat_operand : concat->operands()) {
    const int64_t slice_size =
        concat_operand->shape().dimensions(lhs_contracting_dim);
    std::vector<int64_t> starts(rhs->shape().dimensions().size(), 0);
    std::vector<int64_t> limits(rhs->shape().dimensions().begin(),
                                rhs->shape().dimensions().end());
    std::vector<int64_t> strides(rhs->shape().dimensions().size(), 1);
    std::vector<int64_t> slice_dims(rhs->shape().dimensions().begin(),
                                    rhs->shape().dimensions().end());
    starts[rhs_contracting_dim] = rhs_offset;
    limits[rhs_contracting_dim] = rhs_offset + slice_size;
    slice_dims[rhs_contracting_dim] = slice_size;
    rhs_offset += slice_size;

    Shape rhs_slice_shape =
        ShapeUtil::MakeShape(rhs->shape().element_type(), slice_dims);
    HloInstruction* rhs_slice = computation->AddInstruction(
        HloInstruction::CreateSlice(rhs_slice_shape, rhs, starts, limits,
                                    strides));
    rhs_slice->set_metadata(dot->metadata());

    HloInstruction* partial_dot = computation->AddInstruction(
        HloInstruction::CreateDot(dot->shape(), concat_operand, rhs_slice,
                                  dim_numbers, dot->precision_config()));
    partial_dot->set_metadata(dot->metadata());
    partial_dot->set_frontend_attributes(dot->frontend_attributes());
    partial_dots.push_back(partial_dot);
  }

  if (rhs_offset != rhs->shape().dimensions(rhs_contracting_dim)) {
    return false;
  }

  HloInstruction* replacement = partial_dots[0];
  for (int64_t i = 1; i < partial_dots.size(); ++i) {
    replacement = computation->AddInstruction(HloInstruction::CreateBinary(
        dot->shape(), HloOpcode::kAdd, replacement, partial_dots[i]));
    replacement->set_metadata(dot->metadata());
    replacement->set_frontend_attributes(dot->frontend_attributes());
  }

  LOG(INFO) << "CpuSplitConcatDot rewrote " << dot->name() << " by splitting "
            << concat->name() << " with " << concat->operand_count()
            << " operands";
  TF_RETURN_IF_ERROR(dot->parent()->ReplaceInstruction(dot, replacement));
  return true;
}

}  // namespace cpu
}  // namespace xla
