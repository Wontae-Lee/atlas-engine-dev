#ifndef INCLUDE_ATLAS_MATH_VECTOR_OPERATORS_H
#define INCLUDE_ATLAS_MATH_VECTOR_OPERATORS_H
#include <atlas/math/detail/config.h>
#include <atlas/math/detail/ops.h>
#include <atlas/math/vector/expression.h>
#include <type_traits>
namespace atlas::math {
template <typename T, typename E>
using VectorNeg = VectorUnaryOperator<T, E, detail::Negate<T>>;
template <typename To, typename FromExpr>
using VectorTypeCast = VectorUnaryOperator<To, FromExpr, detail::TypeCast<expr_value_t<FromExpr>, To>>;
template <typename T, typename EL, typename ER>
using VectorAdd = VectorBinaryOperator<T, EL, ER, detail::Add<T>>;
template <typename T, typename EL, typename ER>
using VectorSub = VectorBinaryOperator<T, EL, ER, detail::Sub<T>>;
template <typename T, typename EL, typename ER>
using VectorMul = VectorBinaryOperator<T, EL, ER, detail::Mul<T>>;
template <typename T, typename EL, typename ER>
using VectorDiv = VectorBinaryOperator<T, EL, ER, detail::Div<T>>;
template <typename T, typename E>
using VectorScalarAdd = VectorScalarBinaryOperator<T, E, detail::Add<T>>;
template <typename T, typename E>
using VectorScalarSub = VectorScalarBinaryOperator<T, E, detail::Sub<T>>;
template <typename T, typename E>
using VectorScalarRSub = VectorScalarBinaryOperator<T, E, detail::RSub<T>>;
template <typename T, typename E>
using VectorScalarMul = VectorScalarBinaryOperator<T, E, detail::Mul<T>>;
template <typename T, typename E>
using VectorScalarDiv = VectorScalarBinaryOperator<T, E, detail::Div<T>>;
template <typename T, typename E>
using VectorScalarRDiv = VectorScalarBinaryOperator<T, E, detail::RDiv<T>>;
template <typename EL, typename ER>
using VectorLess = VectorBinaryOperator<bool, EL, ER, detail::Less<expr_value_t<EL>>>;
template <typename EL, typename ER>
using VectorLessEqual = VectorBinaryOperator<bool, EL, ER, detail::LessEqual<expr_value_t<EL>>>;
template <typename EL, typename ER>
using VectorGreater = VectorBinaryOperator<bool, EL, ER, detail::Greater<expr_value_t<EL>>>;
template <typename EL, typename ER>
using VectorGreaterEqual = VectorBinaryOperator<bool, EL, ER, detail::GreaterEqual<expr_value_t<EL>>>;
template <typename EL, typename ER>
using VectorEqual = VectorBinaryOperator<bool, EL, ER, detail::Equal<expr_value_t<EL>>>;
template <typename EL, typename ER>
using VectorNotEqual = VectorBinaryOperator<bool, EL, ER, detail::NotEqual<expr_value_t<EL>>>;
template <typename E>
using VectorScalarLess = VectorScalarBinaryOperator<bool, E, detail::Less<expr_value_t<E>>>;
template <typename E>
using VectorScalarRLess = VectorScalarBinaryOperator<bool, E, detail::RLess<expr_value_t<E>>>;
template <typename E>
using VectorScalarLessEqual = VectorScalarBinaryOperator<bool, E, detail::LessEqual<expr_value_t<E>>>;
template <typename E>
using VectorScalarRLessEqual = VectorScalarBinaryOperator<bool, E, detail::RLessEqual<expr_value_t<E>>>;
template <typename E>
using VectorScalarGreater = VectorScalarBinaryOperator<bool, E, detail::Greater<expr_value_t<E>>>;
template <typename E>
using VectorScalarRGreater = VectorScalarBinaryOperator<bool, E, detail::RGreater<expr_value_t<E>>>;
template <typename E>
using VectorScalarGreaterEqual = VectorScalarBinaryOperator<bool, E, detail::GreaterEqual<expr_value_t<E>>>;
template <typename E>
using VectorScalarRGreaterEqual = VectorScalarBinaryOperator<bool, E, detail::RGreaterEqual<expr_value_t<E>>>;
template <typename E>
using VectorScalarEqual = VectorScalarBinaryOperator<bool, E, detail::Equal<expr_value_t<E>>>;
template <typename E>
using VectorScalarREqual = VectorScalarBinaryOperator<bool, E, detail::REqual<expr_value_t<E>>>;
template <typename E>
using VectorScalarNotEqual = VectorScalarBinaryOperator<bool, E, detail::NotEqual<expr_value_t<E>>>;
template <typename E>
using VectorScalarRNotEqual = VectorScalarBinaryOperator<bool, E, detail::RNotEqual<expr_value_t<E>>>;
template <typename EL, typename ER>
using VectorAnd = VectorBinaryOperator<bool, EL, ER, detail::LogicalAnd<bool>>;
template <typename EL, typename ER>
using VectorOr = VectorBinaryOperator<bool, EL, ER, detail::LogicalOr<bool>>;
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator+(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return VectorAdd<T, EL, ER>(l(), r());
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator+(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarAdd<T, E>(l(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator+(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarAdd<T, E>(r(), s);
}
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return VectorSub<T, EL, ER>(l(), r());
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarSub<T, E>(l(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRSub<T, E>(r(), s);
}
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator*(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return VectorMul<T, EL, ER>(l(), r());
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator*(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarMul<T, E>(l(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator*(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarMul<T, E>(r(), s);
}
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator/(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return VectorDiv<T, EL, ER>(l(), r());
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator/(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarDiv<T, E>(l(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator/(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRDiv<T, E>(r(), s);
}
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const E& e) noexcept {
    using T = expr_value_t<E>;
    return VectorNeg<T, E>(e());
}
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator<(const EL& l, const ER& r) noexcept {
    return VectorLess<EL, ER>(l(), r());
}
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator<=(const EL& l, const ER& r) noexcept {
    return VectorLessEqual<EL, ER>(l(), r());
}
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator>(const EL& l, const ER& r) noexcept {
    return VectorGreater<EL, ER>(l(), r());
}
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator>=(const EL& l, const ER& r) noexcept {
    return VectorGreaterEqual<EL, ER>(l(), r());
}
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator==(const EL& l, const ER& r) noexcept {
    return VectorEqual<EL, ER>(l(), r());
}
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator!=(const EL& l, const ER& r) noexcept {
    return VectorNotEqual<EL, ER>(l(), r());
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator<(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarLess<E>(l(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator<(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRLess<E>(r(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator<=(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarLessEqual<E>(l(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator<=(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRLessEqual<E>(r(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator>(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarGreater<E>(l(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator>(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRGreater<E>(r(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator>=(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarGreaterEqual<E>(l(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator>=(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRGreaterEqual<E>(r(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator==(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarEqual<E>(l(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator==(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarREqual<E>(r(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator!=(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarNotEqual<E>(l(), s);
}
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator!=(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRNotEqual<E>(r(), s);
}
template <typename EL, typename ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator&(const VectorExpression<bool, EL>& l, const VectorExpression<bool, ER>& r) noexcept {
    return VectorAnd<EL, ER>(l(), r());
}
template <typename EL, typename ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator|(const VectorExpression<bool, EL>& l, const VectorExpression<bool, ER>& r) noexcept {
    return VectorOr<EL, ER>(l(), r());
}
}
#endif