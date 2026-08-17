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

#include "xla/service/cpu/runtime_batch_size.h"

#include "xla/executable_run_options.h"

extern "C" int64_t __xla_cpu_runtime_GetBatchSize(
    const void* run_options_ptr) {
  if (run_options_ptr == nullptr) return 0;
  return static_cast<const xla::ExecutableRunOptions*>(run_options_ptr)
      ->batch_size();
}
