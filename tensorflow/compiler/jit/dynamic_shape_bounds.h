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

#ifndef TENSORFLOW_COMPILER_JIT_DYNAMIC_SHAPE_BOUNDS_H_
#define TENSORFLOW_COMPILER_JIT_DYNAMIC_SHAPE_BOUNDS_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_expr.h"

namespace tensorflow {

// Padding the positive root from A to P is safe only if each dimension obeys
// e(A) <= e(P) for every supported A <= P. Conservatively reject expressions
// whose monotonicity cannot be established. Tensor contents are not dimensions.
absl::Status CheckDynamicShapeBounds(absl::Span<const xla::DExpr> expressions,
                                    absl::string_view context);

// Check intermediate and output shapes in every computation, including tuples.
absl::Status CheckDynamicShapeBounds(const xla::HloModuleProto& module);

// Identifies this guard's failure so callers can retry with static arguments.
bool IsDynamicShapeBoundsError(const absl::Status& status);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DYNAMIC_SHAPE_BOUNDS_H_
