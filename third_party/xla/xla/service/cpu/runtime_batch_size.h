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

#ifndef XLA_SERVICE_CPU_RUNTIME_BATCH_SIZE_H_
#define XLA_SERVICE_CPU_RUNTIME_BATCH_SIZE_H_

#include <cstdint>

namespace xla::cpu::runtime {

inline constexpr char kGetBatchSizeSymbolName[] =
    "__xla_cpu_runtime_GetBatchSize";

}  // namespace xla::cpu::runtime

extern "C" int64_t __xla_cpu_runtime_GetBatchSize(
    const void* run_options_ptr);

#endif  // XLA_SERVICE_CPU_RUNTIME_BATCH_SIZE_H_
