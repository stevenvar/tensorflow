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

#include "xla/service/cpu/cpu_concat_fusion.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"

namespace xla::cpu {
namespace {

bool IsSingleUseLoopFusionProducer(const HloInstruction* producer,
                                   const HloInstruction* concat) {
  return producer->opcode() == HloOpcode::kFusion &&
         producer->fusion_kind() == HloInstruction::FusionKind::kLoop &&
         producer->user_count() == 1 && producer->users().front() == concat;
}

absl::StatusOr<bool> TryFuseConcatProducers(HloInstruction* concat) {
  if (concat->opcode() != HloOpcode::kConcatenate ||
      concat->operand_count() < 2 || !concat->shape().IsArray()) {
    return false;
  }

  std::vector<HloInstruction*> producers_to_merge;
  for (HloInstruction* operand : concat->operands()) {
    if (IsSingleUseLoopFusionProducer(operand, concat)) {
      producers_to_merge.push_back(operand);
    }
  }

  if (producers_to_merge.empty()) {
    return false;
  }

  HloComputation* computation = concat->parent();
  const std::string concat_name(concat->name());
  HloInstruction* concat_fusion =
      computation->AddInstruction(HloInstruction::CreateFusion(
          concat->shape(), HloInstruction::FusionKind::kLoop, concat),
                                  absl::StrCat("concat_producer_fusion_",
                                               concat_name));
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(concat, concat_fusion));

  for (HloInstruction* producer : producers_to_merge) {
    concat_fusion->MergeFusionInstruction(producer);
    if (producer->user_count() == 0) {
      TF_RETURN_IF_ERROR(producer->SafelyDropAllControlDependencies());
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(producer));
    }
  }

  LOG(INFO) << "CpuConcatFusion merged " << producers_to_merge.size()
            << " producer fusion(s) into concatenate: " << concat_name;
  return true;
}

}  // namespace

absl::StatusOr<bool> CpuConcatFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> concats;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kConcatenate) {
        concats.push_back(instruction);
      }
    }
  }

  bool changed = false;
  for (HloInstruction* concat : concats) {
    TF_ASSIGN_OR_RETURN(bool rewrote, TryFuseConcatProducers(concat));
    changed |= rewrote;
  }
  return changed;
}

}  // namespace xla::cpu
