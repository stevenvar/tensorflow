#include "tensorflow/core/framework/tensor_shape_expr.h"

#include "xla/printer.h"

namespace tensorflow {

DExpr DExprFromProto(const ExpressionProto& proto) {
  switch (proto.node_type_case()) {
    case ExpressionProto::kConstantValue:
      return DExpr::Const(proto.constant_value());
    case ExpressionProto::kVariableId:
      return DExpr::Var(proto.variable_id());
    case ExpressionProto::kAddNode: {
      const auto& add = proto.add_node();
      return DExprFromProto(add.lhs()) + DExprFromProto(add.rhs());
    }
    case ExpressionProto::kSubNode: {
      const auto& sub = proto.sub_node();
      return DExprFromProto(sub.lhs()) - DExprFromProto(sub.rhs());
    }
    case ExpressionProto::kMulNode: {
      const auto& mul = proto.mul_node();
      return DExprFromProto(mul.lhs()) * DExprFromProto(mul.rhs());
    }
    case ExpressionProto::kDivNode: {
      const auto& div = proto.div_node();
      return DExprFromProto(div.lhs()) / DExprFromProto(div.rhs());
    }
    case ExpressionProto::NODE_TYPE_NOT_SET:
    default:
      return DExpr::Unknown();
  }
}

void ExprToProto(const DExpr& expr, ExpressionProto* proto) {
  if (!expr || expr.is_unknown()) return;

  switch (expr.kind()) {
    case DExpr::Kind::kUnknown:
      return;
    case DExpr::Kind::kConstant:
      proto->set_constant_value(expr->get_val());
      return;
    case DExpr::Kind::kVariable:
      proto->set_variable_id(
          static_cast<const xla::Variable*>(expr.get())->get_id());
      return;
    case DExpr::Kind::kAdd: {
      auto* add = proto->mutable_add_node();
      const auto* node = static_cast<const xla::Add*>(expr.get());
      ExprToProto(DExpr(node->get_lhs()->clone()), add->mutable_lhs());
      ExprToProto(DExpr(node->get_rhs()->clone()), add->mutable_rhs());
      return;
    }
    case DExpr::Kind::kSub: {
      auto* sub = proto->mutable_sub_node();
      const auto* node = static_cast<const xla::Sub*>(expr.get());
      ExprToProto(DExpr(node->get_lhs()->clone()), sub->mutable_lhs());
      ExprToProto(DExpr(node->get_rhs()->clone()), sub->mutable_rhs());
      return;
    }
    case DExpr::Kind::kMul: {
      auto* mul = proto->mutable_mul_node();
      const auto* node = static_cast<const xla::Mul*>(expr.get());
      ExprToProto(DExpr(node->get_lhs()->clone()), mul->mutable_lhs());
      ExprToProto(DExpr(node->get_rhs()->clone()), mul->mutable_rhs());
      return;
    }
    case DExpr::Kind::kDiv: {
      auto* div = proto->mutable_div_node();
      const auto* node = static_cast<const xla::Div*>(expr.get());
      ExprToProto(DExpr(node->get_lhs()->clone()), div->mutable_lhs());
      ExprToProto(DExpr(node->get_rhs()->clone()), div->mutable_rhs());
      return;
    }
  }
}

std::string ExprToString(const DExpr& expr) {
  if (!expr && !expr.is_unknown()) return "";
  xla::StringPrinter printer;
  expr->print(&printer);
  return std::move(printer).ToString();
}

bool ExprEquals(const DExpr& a, const DExpr& b) {
  if (!a || !b) {
    return (!a && !b) && (a.is_unknown() == b.is_unknown());
  }
  return a == b;
}

}  // namespace tensorflow
