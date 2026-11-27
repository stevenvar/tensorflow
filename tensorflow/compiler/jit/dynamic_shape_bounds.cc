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

#include "tensorflow/compiler/jit/dynamic_shape_bounds.h"

#include <utility>
#include <vector>

#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "xla/printer.h"

namespace tensorflow {
namespace {

constexpr char kUnsafeBoundsPayload[] = "tensorflow/unsafe_dynamic_shape_bounds";

struct Growth {
  bool nondecreasing = false;
  bool nonnegative = false;
};

// Increasing negative terms can have a decreasing product, e.g. (A - 100)^2
// below A=100. Products therefore need sign information as well as growth.
Growth GetGrowth(xla::DynExpr* expr) {
  if (expr->kind() == xla::DExprKind::kUnknown) return {};
  if (expr->is_constant()) return {true, expr->get_val() >= 0};
  auto pair = [](auto* op) {
    return std::make_pair(GetGrowth(op->get_lhs()), GetGrowth(op->get_rhs()));
  };
  switch (expr->kind()) {
    case xla::DExprKind::kVariable:
      return {true, true};
    case xla::DExprKind::kAdd: {
      const auto [lhs, rhs] = pair(static_cast<class xla::Add*>(expr));
      return {lhs.nondecreasing && rhs.nondecreasing,
              lhs.nonnegative && rhs.nonnegative};
    }
    case xla::DExprKind::kSub: {
      auto* sub = static_cast<class xla::Sub*>(expr);
      if (!sub->get_rhs()->is_constant()) return {};
      const Growth lhs = GetGrowth(sub->get_lhs());
      return {lhs.nondecreasing,
              lhs.nonnegative && sub->get_rhs()->get_val() <= 0};
    }
    case xla::DExprKind::kMul: {
      auto* mul = static_cast<class xla::Mul*>(expr);
      const auto [lhs, rhs] = pair(mul);
      const bool positive_scale =
          (mul->get_lhs()->is_constant() && lhs.nonnegative) ||
          (mul->get_rhs()->is_constant() && rhs.nonnegative);
      return {lhs.nondecreasing && rhs.nondecreasing &&
                  (positive_scale || (lhs.nonnegative && rhs.nonnegative)),
              lhs.nonnegative && rhs.nonnegative};
    }
    case xla::DExprKind::kDiv: {
      auto* div = static_cast<class xla::Div*>(expr);
      if (!div->get_rhs()->is_constant() || div->get_rhs()->get_val() <= 0) {
        return {};
      }
      return GetGrowth(div->get_lhs());
    }
    case xla::DExprKind::kMax: {
      const auto [lhs, rhs] = pair(static_cast<xla::MaxExpr*>(expr));
      return {lhs.nondecreasing && rhs.nondecreasing,
              lhs.nonnegative || rhs.nonnegative};
    }
    case xla::DExprKind::kGt: {
      auto* gt = static_cast<xla::GtExpr*>(expr);
      return {gt->get_rhs()->is_constant() &&
                  GetGrowth(gt->get_lhs()).nondecreasing,
              true};
    }
    case xla::DExprKind::kSelect: {
      auto* select = static_cast<xla::SelectExpr*>(expr);
      if (!select->get_pred()->is_constant()) return {};
      return GetGrowth(select->get_pred()->get_val() != 0
                           ? select->get_on_true()
                           : select->get_on_false());
    }
    default:
      return {};
  }
}

absl::Status CheckShape(const xla::ShapeProto& shape,
                        absl::string_view context) {
  std::vector<xla::DExpr> expressions;
  expressions.reserve(shape.expressions_size());
  for (const auto& expr : shape.expressions()) {
    expressions.push_back(xla::DExprFromProto(expr));
  }
  auto status = CheckDynamicShapeBounds(expressions, context);
  if (!status.ok()) return status;
  for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
    status = CheckShape(shape.tuple_shapes(i),
                        absl::StrCat(context, " tuple element ", i));
    if (!status.ok()) return status;
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status CheckDynamicShapeBounds(absl::Span<const xla::DExpr> expressions,
                                    absl::string_view context) {
  for (int dim = 0; dim < expressions.size(); ++dim) {
    const auto& expr = expressions[dim];
    if (!expr || !expr->is_dynamic() || GetGrowth(expr.get()).nondecreasing) {
      continue;
    }
    xla::StringPrinter printer;
    expr->print(&printer);
    auto status = absl::FailedPreconditionError(absl::StrCat(
        "Cannot safely pad dynamic shape at ", context, " dimension ", dim,
        ": ", std::move(printer).ToString(),
        " is not proven nondecreasing in the positive dynamic root"));
    status.SetPayload(kUnsafeBoundsPayload, absl::Cord("1"));
    return status;
  }
  return absl::OkStatus();
}

absl::Status CheckDynamicShapeBounds(const xla::HloModuleProto& module) {
  for (const auto& computation : module.computations()) {
    for (const auto& instruction : computation.instructions()) {
      auto status = CheckShape(
          instruction.shape(),
          absl::StrCat(computation.name(), "/", instruction.name()));
      if (!status.ok()) return status;
    }
  }
  return absl::OkStatus();
}

bool IsDynamicShapeBoundsError(const absl::Status& status) {
  return status.GetPayload(kUnsafeBoundsPayload).has_value();
}

}  // namespace tensorflow
