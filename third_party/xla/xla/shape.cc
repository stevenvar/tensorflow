/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/shape.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/printer.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

Constant* AsConstant(DynExpr* expr) {
  return expr != nullptr && expr->kind() == DExpr::Kind::kConstant
             ? static_cast<Constant*>(expr)
             : nullptr;
}

void NormalizeFraction(int64_t* numerator, int64_t* denominator) {
  CHECK(denominator != nullptr);
  CHECK(*denominator != 0);
  int64_t divisor = std::gcd(std::llabs(*numerator), std::llabs(*denominator));
  if (divisor > 1) {
    *numerator /= divisor;
    *denominator /= divisor;
  }
  if (*denominator < 0) {
    *numerator = -*numerator;
    *denominator = -*denominator;
  }
}

// We normalize affine expressions into:
//   (constant + sum_i coeff_i * var_i) / denominator
//
// Concretely, the canonical forms we want to preserve are:
//   p
//   mX
//   mX + p
//   mX + nY
//   (mX) / d
//   (mX + p) / d
//   (mX + nY + p) / d
//
// This lets us combine equivalent trees into one stable representation, e.g.:
//   A/2 + A/2     -> A
//   A/2 + B/2     -> (A + B) / 2
//   4/8 * A       -> A / 2
//   A + (B + 1)   -> A + B + 1   (represented via coefficients + constant)
//
// The important constraint is that we only canonicalize affine expressions over
// an integer denominator. Non-affine expressions stay in tree form and only get
// a minimal local simplification in the fallback path below.
struct CanonicalAffineExpr {
  int64_t denominator = 1;
  int64_t constant = 0;
  std::map<int, int64_t> coefficients;

  bool IsPureConstant() const { return coefficients.empty(); }
};

void NormalizeAffine(CanonicalAffineExpr* expr) {
  CHECK(expr != nullptr);
  CHECK(expr->denominator != 0);
  // Drop zero coefficients so `mX + 0Y + p` becomes `mX + p`.
  for (auto it = expr->coefficients.begin(); it != expr->coefficients.end();) {
    if (it->second == 0) {
      it = expr->coefficients.erase(it);
    } else {
      ++it;
    }
  }
  // Reduce `(mX + nY + p) / d` by the gcd of all integer coefficients.
  int64_t divisor = std::llabs(expr->constant);
  for (const auto& [_, coeff] : expr->coefficients) {
    divisor = std::gcd(divisor, std::llabs(coeff));
  }
  divisor = std::gcd(divisor, std::llabs(expr->denominator));
  if (divisor > 1) {
    expr->constant /= divisor;
    expr->denominator /= divisor;
    for (auto& [_, coeff] : expr->coefficients) {
      coeff /= divisor;
    }
  }
  // Keep the denominator positive so `( -A ) / ( -2 )` canonicalizes to
  // `A / 2` rather than preserving sign in two places.
  if (expr->denominator < 0) {
    expr->denominator = -expr->denominator;
    expr->constant = -expr->constant;
    for (auto& [_, coeff] : expr->coefficients) {
      coeff = -coeff;
    }
  }
  // Canonical zero is just `0`, not `0 / d`.
  if (expr->constant == 0 && expr->coefficients.empty()) {
    expr->denominator = 1;
  }
}

CanonicalAffineExpr MakeConstantAffine(int64_t value) {
  CanonicalAffineExpr expr;
  expr.constant = value;
  return expr;
}

CanonicalAffineExpr MakeVariableAffine(int id) {
  CanonicalAffineExpr expr;
  expr.coefficients[id] = 1;
  return expr;
}

CanonicalAffineExpr AddAffine(const CanonicalAffineExpr& lhs,
                              const CanonicalAffineExpr& rhs, int rhs_sign) {
  CHECK(rhs_sign == 1 || rhs_sign == -1);
  CanonicalAffineExpr result;
  int64_t gcd = std::gcd(lhs.denominator, rhs.denominator);
  int64_t lhs_scale = rhs.denominator / gcd;
  int64_t rhs_scale = lhs.denominator / gcd;
  result.denominator = lhs.denominator * lhs_scale;
  result.constant =
      lhs.constant * lhs_scale + rhs_sign * rhs.constant * rhs_scale;
  for (const auto& [id, coeff] : lhs.coefficients) {
    result.coefficients[id] = coeff * lhs_scale;
  }
  for (const auto& [id, coeff] : rhs.coefficients) {
    result.coefficients[id] += rhs_sign * coeff * rhs_scale;
  }
  NormalizeAffine(&result);
  return result;
}

CanonicalAffineExpr MultiplyAffineByRational(const CanonicalAffineExpr& expr,
                                             int64_t numerator,
                                             int64_t denominator) {
  CHECK(denominator != 0);
  CanonicalAffineExpr result = expr;
  result.constant *= numerator;
  result.denominator *= denominator;
  for (auto& [_, coeff] : result.coefficients) {
    coeff *= numerator;
  }
  NormalizeAffine(&result);
  return result;
}

std::optional<CanonicalAffineExpr> ToCanonicalAffine(const DynExpr* expr) {
  CHECK(expr != nullptr);
  // This recognizes exactly the expression families that fit our affine normal
  // form. If an expression falls outside of that space, we deliberately return
  // std::nullopt and let the fallback simplifier keep the original tree shape.
  switch (expr->kind()) {
    case DExpr::Kind::kUnknown:
      return std::nullopt;
    case DExpr::Kind::kConstant:
      // p -> p
      return MakeConstantAffine(static_cast<const Constant*>(expr)->get_val());
    case DExpr::Kind::kVariable:
      // X -> 1*X
      return MakeVariableAffine(static_cast<const Variable*>(expr)->get_id());
    case DExpr::Kind::kAdd: {
      const auto* add = static_cast<const Add*>(expr);
      auto lhs = ToCanonicalAffine(add->get_lhs());
      auto rhs = ToCanonicalAffine(add->get_rhs());
      if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
      // affine + affine -> affine
      return AddAffine(*lhs, *rhs, /*rhs_sign=*/1);
    }
    case DExpr::Kind::kSub: {
      const auto* sub = static_cast<const Sub*>(expr);
      auto lhs = ToCanonicalAffine(sub->get_lhs());
      auto rhs = ToCanonicalAffine(sub->get_rhs());
      if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
      // affine - affine -> affine
      return AddAffine(*lhs, *rhs, /*rhs_sign=*/-1);
    }
    case DExpr::Kind::kMul: {
      const auto* mul = static_cast<const Mul*>(expr);
      auto lhs = ToCanonicalAffine(mul->get_lhs());
      auto rhs = ToCanonicalAffine(mul->get_rhs());
      if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
      // constant * affine -> affine
      if (lhs->IsPureConstant()) {
        return MultiplyAffineByRational(*rhs, lhs->constant, lhs->denominator);
      }
      // affine * constant -> affine
      if (rhs->IsPureConstant()) {
        return MultiplyAffineByRational(*lhs, rhs->constant, rhs->denominator);
      }
      // affine * affine is not affine in general, so keep it as a tree.
      return std::nullopt;
    }
    case DExpr::Kind::kDiv: {
      const auto* div = static_cast<const Div*>(expr);
      auto lhs = ToCanonicalAffine(div->get_lhs());
      auto rhs = ToCanonicalAffine(div->get_rhs());
      if (!lhs.has_value() || !rhs.has_value() || !rhs->IsPureConstant() ||
          rhs->constant == 0) {
        return std::nullopt;
      }
      // affine / constant -> affine-over-denominator
      return MultiplyAffineByRational(*lhs, rhs->denominator, rhs->constant);
    }
  }
  return std::nullopt;
}

std::unique_ptr<DynExpr> BuildScaledVariableTerm(int id, int64_t coefficient) {
  CHECK(coefficient != 0);
  if (coefficient == 1) {
    return std::make_unique<Variable>(id);
  }
  return std::make_unique<Mul>(DynExpr::_(coefficient), DynExpr::V(id));
}

std::unique_ptr<DynExpr> BuildAffineNumerator(const CanonicalAffineExpr& expr) {
  std::unique_ptr<DynExpr> result;
  // Emit variables in a stable order so `mX + nY` and `nY + mX` rebuild to the
  // same tree shape.
  for (const auto& [id, coeff] : expr.coefficients) {
    auto term = BuildScaledVariableTerm(id, coeff);
    if (result == nullptr) {
      result = std::move(term);
    } else {
      // coeff_i*X_i + coeff_j*X_j + ...
      result = std::make_unique<Add>(result.release(), term.release());
    }
  }
  if (expr.constant != 0 || result == nullptr) {
    auto constant_term = std::make_unique<Constant>(expr.constant);
    if (result == nullptr) {
      // p
      result = std::move(constant_term);
    } else {
      // affine_terms + p
      result = std::make_unique<Add>(result.release(), constant_term.release());
    }
  }
  return result;
}

std::unique_ptr<DynExpr> BuildCanonicalExpr(const CanonicalAffineExpr& expr) {
  CanonicalAffineExpr normalized = expr;
  NormalizeAffine(&normalized);
  auto numerator = BuildAffineNumerator(normalized);
  // Denominator 1 means the affine numerator is already the final canonical
  // form: `p`, `mX`, `mX + p`, `mX + nY`, ...
  if (normalized.denominator == 1) {
    return numerator;
  }
  // Otherwise keep the single shared denominator outside the affine numerator:
  // `(mX + nY + p) / d`.
  return std::make_unique<Div>(numerator.release(),
                               DynExpr::_(normalized.denominator));
}

std::unique_ptr<DynExpr> SimplifyFallback(const DynExpr* expr) {
  // Fallback intentionally does the minimum local cleanup needed to avoid
  // obviously noisy trees. It deliberately does not reassociate sums,
  // distribute division over addition, or otherwise try to invent a more
  // "clever" tree shape.
  switch (expr->kind()) {
    case DExpr::Kind::kUnknown:
    case DExpr::Kind::kConstant:
    case DExpr::Kind::kVariable:
      return expr->clone();
    case DExpr::Kind::kAdd: {
      const auto* add = static_cast<const Add*>(expr);
      auto lhs = std::unique_ptr<DynExpr>(add->get_lhs()->s());
      auto rhs = std::unique_ptr<DynExpr>(add->get_rhs()->s());
      if (lhs->kind() == DExpr::Kind::kUnknown ||
          rhs->kind() == DExpr::Kind::kUnknown) {
        return std::make_unique<UnknownExpr>();
      }
      Constant* l = AsConstant(lhs.get());
      Constant* r = AsConstant(rhs.get());
      // c1 + c2 -> c3
      if (l && r) return std::make_unique<Constant>(l->get_val() + r->get_val());
      // 0 + X -> X
      if (l && l->get_val() == 0) return rhs;
      // X + 0 -> X
      if (r && r->get_val() == 0) return lhs;
      return std::make_unique<Add>(lhs.release(), rhs.release());
    }
    case DExpr::Kind::kSub: {
      const auto* sub = static_cast<const Sub*>(expr);
      auto lhs = std::unique_ptr<DynExpr>(sub->get_lhs()->s());
      auto rhs = std::unique_ptr<DynExpr>(sub->get_rhs()->s());
      if (lhs->kind() == DExpr::Kind::kUnknown ||
          rhs->kind() == DExpr::Kind::kUnknown) {
        return std::make_unique<UnknownExpr>();
      }
      Constant* l = AsConstant(lhs.get());
      Constant* r = AsConstant(rhs.get());
      // c1 - c2 -> c3
      if (l && r) return std::make_unique<Constant>(l->get_val() - r->get_val());
      // X - 0 -> X
      if (r && r->get_val() == 0) return lhs;
      // X - X -> 0
      if (*lhs == *rhs) return std::make_unique<Constant>(0);
      return std::make_unique<Sub>(lhs.release(), rhs.release());
    }
    case DExpr::Kind::kMul: {
      const auto* mul = static_cast<const Mul*>(expr);
      auto lhs = std::unique_ptr<DynExpr>(mul->get_lhs()->s());
      auto rhs = std::unique_ptr<DynExpr>(mul->get_rhs()->s());
      if (lhs->kind() == DExpr::Kind::kUnknown ||
          rhs->kind() == DExpr::Kind::kUnknown) {
        return std::make_unique<UnknownExpr>();
      }
      Constant* l = AsConstant(lhs.get());
      Constant* r = AsConstant(rhs.get());
      // c1 * c2 -> c3
      if (l && r) return std::make_unique<Constant>(l->get_val() * r->get_val());
      // 0 * X -> 0 and X * 0 -> 0
      if ((l && l->get_val() == 0) || (r && r->get_val() == 0)) {
        return std::make_unique<Constant>(0);
      }
      // 1 * X -> X
      if (l && l->get_val() == 1) return rhs;
      // X * 1 -> X
      if (r && r->get_val() == 1) return lhs;
      // Keep constants on the left for a stable fallback tree shape.
      if (r != nullptr) std::swap(lhs, rhs);
      return std::make_unique<Mul>(lhs.release(), rhs.release());
    }
    case DExpr::Kind::kDiv: {
      const auto* div = static_cast<const Div*>(expr);
      auto lhs = std::unique_ptr<DynExpr>(div->get_lhs()->s());
      auto rhs = std::unique_ptr<DynExpr>(div->get_rhs()->s());
      if (lhs->kind() == DExpr::Kind::kUnknown ||
          rhs->kind() == DExpr::Kind::kUnknown) {
        return std::make_unique<UnknownExpr>();
      }
      Constant* l = AsConstant(lhs.get());
      Constant* r = AsConstant(rhs.get());
      // 0 / X -> 0 only when the denominator is proven nonzero.
      if (l && l->get_val() == 0 && r && r->get_val() != 0) {
        return std::make_unique<Constant>(0);
      }
      // X / 1 -> X
      if (r && r->get_val() == 1) return lhs;
      // c1 / c2 -> reduced constant or reduced rational literal when safe
      if (l && r && r->get_val() != 0) {
        int64_t numerator = l->get_val();
        int64_t denominator = r->get_val();
        NormalizeFraction(&numerator, &denominator);
        if (denominator == 1) return std::make_unique<Constant>(numerator);
        return std::make_unique<Div>(DynExpr::_(numerator),
                                     DynExpr::_(denominator));
      }
      return std::make_unique<Div>(lhs.release(), rhs.release());
    }
  }
  return expr->clone();
}

std::unique_ptr<DynExpr> SimplifyCanonical(const DynExpr* expr) {
  if (expr->kind() == DExpr::Kind::kUnknown) {
    return std::make_unique<UnknownExpr>();
  }
  // Prefer the affine normal form whenever possible so simplify() produces one
  // stable canonical tree instead of a collection of equivalent trees.
  if (auto canonical = ToCanonicalAffine(expr); canonical.has_value()) {
    // constant / variable / affine sum / affine-over-denominator
    return BuildCanonicalExpr(*canonical);
  }
  // Non-affine trees keep their overall structure and only get minimal cleanup.
  return SimplifyFallback(expr);
}

}  // namespace

const DExpr& Shape::MissingExpression() {
  static const DExpr missing = DExpr::Unknown(kMissingExpressionSentinel);
  return missing;
}

DynExpr* operator*(DynExpr& lhs, DynExpr& rhs) {
  return new Mul(lhs.clone().release(), rhs.clone().release());
}
DynExpr* operator*(int64_t k, DynExpr& rhs) {
  return new Mul(DynExpr::_(k), rhs.clone().release());
}
DynExpr* operator/(DynExpr& lhs, DynExpr& rhs) {
  return new Div(lhs.clone().release(), rhs.clone().release());
}
DynExpr* operator/(DynExpr& lhs, int64_t d) {
  return new Div(lhs.clone().release(), DynExpr::_(d));
}
DynExpr* operator+(DynExpr& lhs, DynExpr& rhs) {
  return new Add(lhs.clone().release(), rhs.clone().release());
}
DynExpr* operator+(DynExpr& lhs, int64_t d) {
  return new Add(lhs.clone().release(), DynExpr::_(d));
}
DynExpr* operator-(DynExpr& lhs, DynExpr& rhs) {
  return new Sub(lhs.clone().release(), rhs.clone().release());
}
DynExpr* operator-(DynExpr& lhs, int64_t d) {
  return new Sub(lhs.clone().release(), DynExpr::_(d));
}
bool operator==(DynExpr& lhs, DynExpr& rhs) {
  return DynExpr::equal(&lhs, &rhs);
}
bool operator==(DynExpr& lhs, int64_t d) {
  auto rhs = std::unique_ptr<DynExpr>(DynExpr::_(d));
  return DynExpr::equal(&lhs, rhs.get());
}
bool operator<(DynExpr& lhs, int64_t d) {
  return lhs.is_constant() && lhs.get_val() < d;
}

bool DynExpr::equal(DynExpr* expr1, DynExpr* expr2) {
  auto e1 = std::unique_ptr<DynExpr>(expr1->s());
  auto e2 = std::unique_ptr<DynExpr>(expr2->s());
  if (e1 == nullptr || e2 == nullptr) return false;
  if (e1->kind() == DExpr::Kind::kConstant &&
      e2->kind() == DExpr::Kind::kConstant) {
    return static_cast<Constant*>(e1.get())->get_val() ==
           static_cast<Constant*>(e2.get())->get_val();
  }
  // Var x = Var y <=> x = y
  if (e1->kind() == DExpr::Kind::kVariable &&
      e2->kind() == DExpr::Kind::kVariable) {
    return static_cast<Variable*>(e1.get())->get_id() ==
           static_cast<Variable*>(e2.get())->get_id();
  }
  if (e1->kind() == DExpr::Kind::kUnknown &&
      e2->kind() == DExpr::Kind::kUnknown) {
    int lhs_id = static_cast<UnknownExpr*>(e1.get())->get_id();
    int rhs_id = static_cast<UnknownExpr*>(e2.get())->get_id();
    return lhs_id != 0 && lhs_id == rhs_id;
  }
  // a * b = c * d <=> (a = c /\ b = d) \/ (a = d /\ b = c)
  if (e1->kind() == DExpr::Kind::kMul &&
      e2->kind() == DExpr::Kind::kMul) {
    auto* ab = static_cast<Mul*>(e1.get());
    auto* cd = static_cast<Mul*>(e2.get());
    auto a = ab->get_lhs();
    auto b = ab->get_rhs();
    auto c = cd->get_lhs();
    auto d = cd->get_rhs();
    return (*a == *c && *b == *d) || (*a == *d && *b == *c);
  }
  // a / b = c / d <=> (a = c /\ b = d)
  if (e1->kind() == DExpr::Kind::kDiv &&
      e2->kind() == DExpr::Kind::kDiv) {
    auto* ab = static_cast<Div*>(e1.get());
    auto* cd = static_cast<Div*>(e2.get());
    auto a = ab->get_lhs();
    auto b = ab->get_rhs();
    auto c = cd->get_lhs();
    auto d = cd->get_rhs();
    return *a == *c && *b == *d;
  }
  // a + b = c + d <=> (a = c /\ b = d) \/ (a = d /\ b = c)
  if (e1->kind() == DExpr::Kind::kAdd &&
      e2->kind() == DExpr::Kind::kAdd) {
    auto* ab = static_cast<Add*>(e1.get());
    auto* cd = static_cast<Add*>(e2.get());
    auto a = ab->get_lhs();
    auto b = ab->get_rhs();
    auto c = cd->get_lhs();
    auto d = cd->get_rhs();
    return (*a == *c && *b == *d) || (*a == *d && *b == *c);
  }
  // a - b = c - d <=> (a = c /\ b = d)
  if (e1->kind() == DExpr::Kind::kSub &&
      e2->kind() == DExpr::Kind::kSub) {
    auto* ab = static_cast<Sub*>(e1.get());
    auto* cd = static_cast<Sub*>(e2.get());
    auto* a = ab->get_lhs();
    auto* b = ab->get_rhs();
    auto* c = cd->get_lhs();
    auto* d = cd->get_rhs();
    return *a == *c && *b == *d;
  }
  return false;
}

// Simplification methods
DynExpr* Constant::s() { return SimplifyCanonical(this).release(); }

DynExpr* Variable::s() { return SimplifyCanonical(this).release(); }

DynExpr* Mul::s() { return SimplifyCanonical(this).release(); }

DynExpr* Add::s() { return SimplifyCanonical(this).release(); }

DynExpr* Sub::s() { return SimplifyCanonical(this).release(); }

DynExpr* Div::s() { return SimplifyCanonical(this).release(); }

std::ostream& operator<<(std::ostream& os, DynExpr* expr) {
  auto simplified = std::unique_ptr<DynExpr>(expr->s());
  StringPrinter printer;
  simplified->print(&printer);
  os << std::move(printer).ToString();
  return os;
}

DynExpr* DynExpr::zero = new Constant(0);
DynExpr* DynExpr::one = new Constant(1);

// Defined in .cc file to avoid inlining these large routines
Shape::Shape() = default;
Shape::~Shape() = default;
Shape::Shape(const Shape&) = default;
Shape::Shape(Shape&&) noexcept = default;
Shape& Shape::operator=(const Shape&) = default;
Shape& Shape::operator=(Shape&&) noexcept = default;

Shape::Shape(const PrimitiveType element_type) {
  CHECK(element_type == TOKEN || element_type == OPAQUE_TYPE ||
        element_type == BUFFER)
      << "Invalid element type for token or opaque shape: " << element_type_;
  set_element_type(element_type);
}

Shape::Shape(const PrimitiveType element_type,
             const absl::Span<const int64_t> dimensions,
             const absl::Span<const bool> dynamic_dimensions) {
  CHECK(primitive_util::IsArrayType(element_type))
      << "Invalid element type for array shape: " << element_type;
  if (!dynamic_dimensions.empty()) {
    CHECK_EQ(dimensions.size(), dynamic_dimensions.size())
        << "If dynamic_dimensions is provided, it must have the same size as "
           "dimensions.";
  }

  set_element_type(element_type);
  auto& state = array_state();
  state.dimensions = {dimensions.begin(), dimensions.end()};
  if (dynamic_dimensions.empty()) {
    // Assume all dimensions are static.
    state.dynamic_dimensions.resize(dimensions.size(), false);
  } else {
    state.dynamic_dimensions = absl::InlinedVector<bool, InlineRank()>(
        dynamic_dimensions.begin(), dynamic_dimensions.end());
  }
}

Shape::Shape(std::vector<Shape> tuple_shapes) {
  set_element_type(TUPLE);
  tuple_state().tuple_shapes = std::move(tuple_shapes);
}

/* static */ Shape Shape::MakeBufferShape(Shape element_shape) {
  CHECK(element_shape.IsArray())
      << "element_shape must be an array shape to create a buffer shape.";
  Shape shape(BUFFER);
  shape.buffer_state().buffer_shape = {std::move(element_shape)};
  return shape;
}

Shape::Shape(const ShapeProto& shape_proto) {
  *this = FromProto(shape_proto).value_or(Shape());
}

absl::StatusOr<Shape> Shape::FromProto(const ShapeProto& shape_proto) {

  // LOG(INFO) << "FROM PROTO:\n" << shape_proto.DebugString() << std::endl;

  Shape shape;
  shape.set_element_type(shape_proto.element_type());
  if (auto* const state = shape.if_array_state()) {
    const int num_dims = shape_proto.dimensions_size();
    const int num_is_dynamic_dims = shape_proto.is_dynamic_dimension_size();
    const int num_expressions = shape_proto.expressions_size();
    state->dimensions.reserve(num_dims);
    state->dynamic_dimensions.reserve(num_dims);
    state->expressions.reserve(num_dims);
    if (num_is_dynamic_dims != 0) {
      TF_RET_CHECK(num_dims == num_is_dynamic_dims)
          << "Malformed shape proto: number of is_dynamic_dimension "
             "fields ("
          << num_is_dynamic_dims << ") does not match number of dimension "
          << "fields (" << num_dims << ").";
    }
    if (num_expressions != 0) {
      TF_RET_CHECK(num_dims == num_expressions)
          << "Malformed shape proto: number of expressions "
             "fields ("
          << num_expressions << ") does not match number of dimension "
          << "fields (" << num_dims << ").";
    }
    for (int i = 0; i < num_dims; ++i) {
      const bool is_dynamic =
          (i < num_is_dynamic_dims) && shape_proto.is_dynamic_dimension(i);
      // We don't want to crash due to a malformed proto, so use
      // UnsafeAddDimension. We expect that the caller will eventually call a
      // validation routine that will detect the error in case the dimension
      // value is invalid.
      DExpr expression =
          (i < num_expressions) ? DExprFromProto(shape_proto.expressions(i))
                                : DExpr::Const(shape_proto.dimensions(i));
      shape.UnsafeAddDimension(shape_proto.dimensions(i), is_dynamic,
                               expression);
    }
  } else if (auto* const state = shape.if_tuple_state()) {
    state->tuple_shapes.reserve(shape_proto.tuple_shapes_size());
    for (const ShapeProto& element_shape : shape_proto.tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(Shape tuple_shape, Shape::FromProto(element_shape));
      state->tuple_shapes.emplace_back(std::move(tuple_shape));
    }
  } else if (auto* const state = shape.if_buffer_state()) {
    state->buffer_shape.emplace_back(shape_proto.tuple_shapes(0));
  }
  if (shape_proto.has_layout()) {
    TF_RET_CHECK(shape.IsArray()) << "Malformed shape proto: element_type "
                                  << PrimitiveType_Name(shape.element_type())
                                  << " should not have a layout.";
    TF_ASSIGN_OR_RETURN(*shape.mutable_layout(),
                        Layout::FromProto(shape_proto.layout()));
  }
  // LOG(INFO) << "FROM PROTO " << shape << "\n";
  return shape;
}

ShapeProto Shape::ToProto() const {
  ShapeProto proto;
  proto.set_element_type(element_type_);

  // LOG(INFO) << "TO PROTO " << ToString() << "\n";

  if (const auto* const state = if_array_state()) {
    proto.mutable_dimensions()->Reserve(state->dimensions.size());
    for (const int64_t dimension : state->dimensions) {
      proto.add_dimensions(dimension);
    }
    for (const bool dynamic : state->dynamic_dimensions) {
      proto.add_is_dynamic_dimension(dynamic);
    }
    for (const DExpr& e : state->expressions) {
      ExpressionProto* eproto = proto.add_expressions();
      CHECK(e.get() != nullptr) << "Missing expression in expression list.";
      e->to_proto(eproto);
    }
    if (state->layout.has_value()) {
      *proto.mutable_layout() = state->layout->ToProto();
    }
  } else if (const auto* const state = if_tuple_state()) {
    proto.mutable_tuple_shapes()->Reserve(state->tuple_shapes.size());
    for (const Shape& shape : state->tuple_shapes) {
      *proto.add_tuple_shapes() = shape.ToProto();
    }
  } else if (const auto* const state = if_buffer_state()) {
    proto.mutable_tuple_shapes()->Reserve(1);
    *proto.add_tuple_shapes() = state->buffer_shape[0].ToProto();
  }
  // LOG(INFO) << "DEBUG VIEW:\n" << proto.DebugString() << std::endl;
  return proto;
}

const Shape::ArrayState& Shape::array_state() const {
  const auto* const state = if_array_state();
  CHECK(state) << "Expected an array shape. Got " << ToString()
               << "\nThis is a programmer error. Please read "
                  "the Shape object's array properties (e.g. dimensions) "
                  "only when it's an array shape.";
  return *state;
}

Shape::ArrayState& Shape::array_state() {
  auto* const state = if_array_state();
  CHECK(state) << "Expected an array shape. Got " << ToString()
               << "\nThis is a programmer error. Please mutate "
                  "the Shape object's array properties (e.g. dimensions) "
                  "only when it's an array shape.";
  return *state;
}

const Shape::TupleState& Shape::tuple_state() const {
  const auto* const state = if_tuple_state();
  CHECK(state) << "Expected a tuple shape. Got " << ToString()
               << "\nThis is a programmer error. Please read "
                  "the Shape object's tuple properties (e.g. tuple_shapes) "
                  "only when it's a tuple shape.";
  return *state;
}

Shape::TupleState& Shape::tuple_state() {
  auto* const state = if_tuple_state();
  CHECK(state) << "Expected a tuple shape. Got " << ToString()
               << "\nThis is a programmer error. Please mutate "
                  "the Shape object's tuple properties (e.g. tuple_shapes) "
                  "only when it's a tuple shape.";
  return *state;
}

const Shape::BufferState& Shape::buffer_state() const {
  const auto* const state = if_buffer_state();
  CHECK(state) << "Expected a buffer shape. Got " << ToString()
               << "\nThis is a programmer error. Please read "
                  "the Shape object's buffer properties (e.g. buffer_shapes) "
                  "only when it's a buffer shape.";
  return *state;
}

Shape::BufferState& Shape::buffer_state() {
  auto* const state = if_buffer_state();
  CHECK(state) << "Expected a buffer shape. Got " << ToString()
               << "\nThis is a programmer error. Please mutate "
                  "the Shape object's buffer properties (e.g. buffer_shapes) "
                  "only when it's a buffer shape.";
  return *state;
}

void Shape::Print(Printer* printer, bool print_layout) const {
  if (print_layout) {
    ShapeUtil::PrintHumanStringWithLayout(printer, *this);
  } else {
    ShapeUtil::PrintHumanString(printer, *this);
  }
}

std::string Shape::ToString(bool print_layout) const {
  if (print_layout) {
    return ShapeUtil::HumanStringWithLayout(*this);
  } else {
    return ShapeUtil::HumanString(*this);
  }
}

bool Shape::AreAllLeavesIntegers() const {
  if (const auto* const state = if_tuple_state()) {
    return absl::c_all_of(state->tuple_shapes, [](const Shape& s) {
      return s.AreAllLeavesIntegers();
    });
  }
  return primitive_util::IsIntegralType(element_type());
}

void Shape::add_dimensions(int64_t value, bool is_dynamic, DExpr expr) {
  if (value < 0) {
    CHECK(is_dynamic) << "static dimension must have size >= 0 instead of "
                      << value << ".";
    CHECK_EQ(value, kUnboundedSize)
        << "dynamic dimension must have size == kUnboundedSize or >= 0.";
  }
  UnsafeAddDimension(
      value, is_dynamic,
      expr ? std::move(expr) : DExpr::Const(value));
}

void Shape::set_dynamic_dimension(int dimension, bool is_dynamic) {
  auto& state = array_state();
  // Ensure that the dimension size is valid for the new dynamic-ness.
  CheckDimensionSize(dimension, state.dimensions[dimension], is_dynamic);
  state.dynamic_dimensions[dimension] = is_dynamic;
}

void Shape::set_expression(int dimension, DExpr e) {
  auto& state = array_state();
  state.expressions[dimension] =
      e ? std::move(e) : DExpr::Const(state.dimensions[dimension]);
}

void Shape::set_expressions(std::vector<DExpr> exps) {
  auto& state = array_state();
  CHECK_LE(exps.size(), state.dimensions.size());
  state.expressions.resize(state.dimensions.size());
  for (size_t i = 0; i < state.dimensions.size(); ++i) {
    state.expressions[i] = i < exps.size()
                               ? std::move(exps[i])
                               : DExpr::Const(state.dimensions[i]);
    if (!state.expressions[i]) {
      state.expressions[i] = DExpr::Const(state.dimensions[i]);
    }
  }
}

void Shape::set_dimensions(int index, int64_t size,
                           std::optional<bool> is_dynamic) {
  auto& state = array_state();
  const bool dynamic =
      is_dynamic.has_value() ? *is_dynamic : state.dynamic_dimensions[index];
  CheckDimensionSize(index, size, dynamic);
  state.dimensions[index] = size;
  state.dynamic_dimensions[index] = dynamic;
  state.expressions[index] = DExpr::Const(size);
}

void Shape::set_dimensions_minor(int index, int64_t size,
                                 std::optional<bool> is_dynamic) {
  const int physical_index = layout().minor_to_major(index);
  set_dimensions(physical_index, size, is_dynamic);
}

void Shape::CheckDimensionSize(int dim_index, int64_t size, bool is_dynamic) {
  if (is_dynamic) {
    if (size < 0) {
      CHECK_EQ(size, kUnboundedSize) << "the " << dim_index
                                     << "-th dimension is dynamic and must "
                                        "have size == kUnboundedSize or >= 0.";
    }
  } else {
    CHECK_GE(size, 0) << "the " << dim_index
                      << "-th dimension is static and must have size >= 0.";
  }
}

void Shape::UnsafeAddDimension(int64_t value, bool is_dynamic, DExpr exp) {
  auto& state = array_state();
  CHECK_EQ(state.dimensions.size(), state.dynamic_dimensions.size())
      << "where the shape is " << ToString();
  CHECK_EQ(state.dimensions.size(), state.expressions.size())
      << "where the shape is " << ToString();
  state.dimensions.push_back(value);
  state.dynamic_dimensions.push_back(is_dynamic);
  state.expressions.push_back(exp ? std::move(exp) : DExpr::Const(value));
}

bool Shape::is_static() const {
  if (const auto* const state = if_tuple_state()) {
    return absl::c_all_of(state->tuple_shapes,
                          [](const Shape& s) { return s.is_static(); });
  }
  if (const auto* const state = if_array_state()) {
    return !absl::c_any_of(state->dynamic_dimensions, [](bool b) { return b; });
  }
  return true;
}

bool Shape::is_unbounded_dynamic() const {
  if (const auto* const state = if_tuple_state()) {
    return absl::c_any_of(state->tuple_shapes, [](const Shape& subshape) {
      return subshape.is_unbounded_dynamic();
    });
  }
  if (const auto* const state = if_array_state()) {
    return absl::c_any_of(state->dimensions,
                          [](int64_t dim) { return dim == kUnboundedSize; });
  }
  return false;
}

bool Shape::is_bounded_dynamic() const {
  if (const auto* const state = if_tuple_state()) {
    return absl::c_any_of(state->tuple_shapes, [](const Shape& subshape) {
      return subshape.is_bounded_dynamic();
    });
  }
  if (const auto* const state = if_array_state()) {
    for (auto i = 0; i < state->dimensions.size(); ++i) {
      if (is_bounded_dynamic_dimension(i)) return true;
    }
    return false;
  }
  return false;
}

void Shape::DeleteDimension(int64_t dim_to_delete) {
  auto& state = array_state();
  CHECK_GE(dim_to_delete, 0);
  CHECK_LT(dim_to_delete, state.dimensions.size());
  state.dimensions.erase(state.dimensions.begin() + dim_to_delete);
  state.dynamic_dimensions.erase(state.dynamic_dimensions.begin() +
                                 dim_to_delete);
  state.expressions.erase(state.expressions.begin() +
                                 dim_to_delete);
  if (LayoutUtil::HasLayout(*this)) {
    state.layout->DeleteDimension(dim_to_delete);  // NOLINT: optional-access
  }
}

void Shape::DeleteDimensions(absl::Span<const int64_t> dims_to_delete) {
  auto& state = array_state();
  std::vector<int64_t> sorted_dims_to_delete(dims_to_delete.begin(),
                                             dims_to_delete.end());
  absl::c_sort(sorted_dims_to_delete);
  state.dimensions = RemoveElements(sorted_dims_to_delete, state.dimensions);
  state.dynamic_dimensions =
      RemoveElements(sorted_dims_to_delete, state.dynamic_dimensions);
  state.expressions =
      RemoveElements(sorted_dims_to_delete, state.expressions);
  if (LayoutUtil::HasLayout(*this)) {
    for (auto it = sorted_dims_to_delete.rbegin();
         it != sorted_dims_to_delete.rend(); ++it) {
      state.layout->DeleteDimension(*it);  // NOLINT: optional-access
    }
  }
}

void Shape::CheckStateIsEmpty() const {
  if (const auto* const state = if_array_state()) {
    CHECK(state->dimensions.empty()) << ToString();
    CHECK(state->dynamic_dimensions.empty()) << ToString();
    CHECK(state->expressions.empty()) << ToString();
    CHECK(!state->layout.has_value()) << ToString();
  } else if (const auto* const state = if_tuple_state()) {
    CHECK(state->tuple_shapes.empty()) << ToString();
  }
}

const std::vector<Shape>& Shape::tuple_shapes() const {
  return tuple_state().tuple_shapes;
}

const Shape& Shape::buffer_shape() const {
  return buffer_state().buffer_shape[0];
}

void Shape::Clear() {
  // Before setting the element type to invalid, we need to clear the state
  // because the state may be non-empty if the shape was previously valid.
  // Without this step, set_element_type() may CHECK-fail.
  if (auto* const state = if_array_state()) {
    *state = ArrayState();
  } else if (auto* const state = if_tuple_state()) {
    *state = TupleState();
  }
  set_element_type(PRIMITIVE_TYPE_INVALID);
}

void Shape::set_element_type(const PrimitiveType value) {
  element_type_ = value;

  // Make sure the variant state matches the element type.
  // If we have to change the case of the variant, and the current case is not
  // empty, it's likely a programmer error - we CHECK-fail to catch it.
  if (element_type_ == TOKEN) {
    if (!if_token_state()) {
      CheckStateIsEmpty();
      state_ = TokenState();
    }
    return;
  }
  if (element_type_ == OPAQUE_TYPE) {
    if (!if_opaque_state()) {
      CheckStateIsEmpty();
      state_ = OpaqueState();
    }
    return;
  }
  if (element_type_ == TUPLE) {
    if (!if_tuple_state()) {
      CheckStateIsEmpty();
      state_ = TupleState();
    }
    return;
  }
  if (element_type_ == BUFFER) {
    if (!if_buffer_state()) {
      CheckStateIsEmpty();
      state_ = BufferState();
    }
    return;
  }
  if (primitive_util::IsArrayType(element_type_)) {
    if (!if_array_state()) {
      CheckStateIsEmpty();
      state_ = ArrayState();
    }
    return;
  }
  // Treat all other types as invalid.
  if (element_type_ != PRIMITIVE_TYPE_INVALID) {
    LOG(ERROR) << "Unsupported element type: " << element_type_;
    element_type_ = PRIMITIVE_TYPE_INVALID;
  }
  if (!if_invalid_state()) {
    CheckStateIsEmpty();
    state_ = InvalidState();
  }
}

const Shape& Shape::tuple_shapes(int index) const {
  return tuple_state().tuple_shapes[index];
}

Shape* Shape::add_tuple_shapes() {
  auto& state = tuple_state();
  state.tuple_shapes.push_back(Shape());
  return &state.tuple_shapes.back();
}

bool Shape::Equal::operator()(const Shape& lhs, const Shape& rhs) {
  if (lhs.IsTuple()) {
    return rhs.IsTuple() &&
           absl::c_equal(
               lhs.tuple_shapes(), rhs.tuple_shapes(),
               [=](const Shape& l, const Shape& r) { return (*this)(l, r); });
  }
  if (lhs.IsBuffer() || rhs.IsBuffer()) {
    if (!ignore_buffer_) {
      return lhs.IsBuffer() && rhs.IsBuffer() &&
             lhs.buffer_shape() == rhs.buffer_shape();
    }
    auto underline_shape = [](const Shape& shape) {
      if (shape.IsBuffer()) {
        return shape.buffer_shape();
      }
      return shape;
    };
    return underline_shape(lhs) == underline_shape(rhs);
  }

  if (!lhs.IsArray()) {
    // Non-tuple, non-array tupes such as opaque and token types are trivially
    // the same.
    return lhs.element_type() == rhs.element_type();
  }

  if (!rhs.IsArray()) {
    return false;
  }

  if (!ignore_element_type_) {
    if ((ignore_fp_precision_ &&
         !ShapeUtil::SameElementTypeIgnoringFpPrecision(lhs, rhs)) ||
        (!ignore_fp_precision_ && !ShapeUtil::SameElementType(lhs, rhs))) {
      VLOG(3) << "CompareShapes: lhs element type != rhs element type";
      return false;
    }
  }

  if (!ignore_dimensions_) {
    if (!ShapeUtil::SameRank(lhs, rhs)) {
      VLOG(3) << "CompareShapes: lhs rank != rhs rank";
      return false;
    }
    for (int i = 0; i < lhs.dimensions().size(); ++i) {
      if (ignore_dynamic_dimension_ &&
          (lhs.is_unbounded_dynamic_dimension(i) ||
           rhs.is_unbounded_dynamic_dimension(i))) {
        continue;
      }
      if (i == 0 && ignore_batch_ &&
          (lhs.outer_multiplier() > 0 || rhs.outer_multiplier() > 0)) {
        VLOG(3) << "CompareShapes: batch dimension found. Forcely compatible";
        continue;
      }
      if (lhs.dimensions(i) != rhs.dimensions(i)) {
        VLOG(3) << "CompareShapes: lhs dimensions != rhs dimensions";
        return false;
      }
    }
  } else {
    if (!ShapeUtil::SameRank(lhs, rhs)) {
      VLOG(3) << "CompareShapes: lhs rank != rhs rank";
      return false;
    }
  }

  if (!ignore_layout_) {
    if (lhs.IsArray()) {
      Layout::Equal equal;
      if (lhs.has_layout() || rhs.has_layout()) {
        if (!lhs.has_layout() || !rhs.has_layout()) {
          VLOG(3) << "CompareShapes: both shapes do not have layouts";
          return false;
        }
        if (ignore_tiles_in_layout_) {
          equal.IgnoreTiles();
        }
        if (ignore_element_size_in_layout_) {
          equal.IgnoreElementSize();
        }
        if (ignore_memory_space_in_layout_) {
          equal.IgnoreMemorySpace();
        }
        if (ignore_tail_padding_alignment_in_elements_in_layout_) {
          equal.IgnoreTailPaddingAlignmentInElements();
        }
        if (ignore_split_config_in_layout_) {
          equal.IgnoreSplitConfigs();
        }
        if (!equal(lhs.layout(), rhs.layout())) {
          VLOG(3) << "CompareShapes: lhs layout != rhs layout";
          return false;
        }
      }
    }
  }

  if (!ignore_dynamic_dimension_) {
    for (int i = 0; i < lhs.dimensions().size(); ++i) {
      if (lhs.is_dynamic_dimension(i) != rhs.is_dynamic_dimension(i)) {
        VLOG(3) << "CompareShapes: lhs and rhs have different dynamic "
                   "dimensions.";
        return false;
      }
    }
  }
  return true;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  out << shape.ToString(/*print_layout=*/true);
  return out;
}

ProgramShape::ProgramShape() = default;
ProgramShape::~ProgramShape() = default;
ProgramShape::ProgramShape(const ProgramShape&) = default;
ProgramShape::ProgramShape(ProgramShape&&) = default;
ProgramShape& ProgramShape::operator=(const ProgramShape&) = default;
ProgramShape& ProgramShape::operator=(ProgramShape&&) = default;

ProgramShape::ProgramShape(const ProgramShapeProto& program_shape_proto) {
  auto program_shape = FromProto(program_shape_proto);
  if (!program_shape.ok()) {
    LOG(ERROR) << "Failed to parse ProgramShapeProto: "
               << program_shape_proto.DebugString();
    return;
  }
  *this = std::move(*program_shape);
}

absl::StatusOr<ProgramShape> ProgramShape::FromProto(
    const ProgramShapeProto& program_shape_proto) {
  ProgramShape program_shape;
  const int num_params = program_shape_proto.parameters_size();
  const int num_param_names = program_shape_proto.parameter_names_size();
  TF_RET_CHECK(num_params == num_param_names)
      << "ProgramShapeProto has different numbers of parameters and "
         "parameter names: "
      << num_params << " vs " << num_param_names;
  program_shape.parameters_.reserve(num_params);
  program_shape.parameter_names_.reserve(num_params);
  for (int i = 0; i < num_params; ++i) {
    const std::string& name =
        i < num_param_names ? program_shape_proto.parameter_names(i) : "";
    TF_ASSIGN_OR_RETURN(Shape shape,
                        Shape::FromProto(program_shape_proto.parameters(i)));
    program_shape.AddParameter(shape, name);
  }
  TF_ASSIGN_OR_RETURN(*program_shape.mutable_result(),
                      Shape::FromProto(program_shape_proto.result()));
  return program_shape;
}

ProgramShapeProto ProgramShape::ToProto() const {
  ProgramShapeProto proto;
  for (const Shape& shape : parameters()) {
    *proto.add_parameters() = shape.ToProto();
  }
  *proto.mutable_result() = result().ToProto();
  for (const std::string& name : parameter_names()) {
    proto.add_parameter_names(name);
  }
  return proto;
}

void ProgramShape::Print(Printer* printer) const {
  ShapeUtil::PrintHumanString(printer, *this);
}

std::string ProgramShape::ToString() const {
  return ShapeUtil::HumanString(*this);
}

std::ostream& operator<<(std::ostream& out, const ProgramShape& program_shape) {
  out << program_shape.ToString() << "\n";
  return out;
}

}  // namespace xla
