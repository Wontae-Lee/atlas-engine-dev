#pragma once

#include <atlas/math/detail/ops.h>
#include <atlas/math/vector/vector_expression.h>
#include <type_traits>

namespace atlas::math {

/**
 * @file operators.h
 * @brief Operator overloads and shorthand aliases for vector expression templates.
 *
 * @details
 * This header defines:
 * 1) Type aliases (`VectorAdd`, `VectorMul`, ...) that bind expression-node templates
 *    (`VectorUnaryOperator`, `VectorBinaryOperator`, `VectorScalarBinaryOperator`) to
 *    specific operator functors from `atlas::math::detail`.
 * 2) Free-function operator overloads (`+`, `-`, `*`, `/`, comparisons, logical ops)
 *    that build **lazy expression nodes** instead of performing immediate computation.
 *
 * The design goal is to allow natural math syntax while preserving expression-template
 * fusion (single-pass evaluation / single kernel launch in GPU backends).
 *
 * @note
 * - These overloads are intended for expression types satisfying `VectorExpressionType`.
 * - Most overloads return lightweight nodes holding const references to their operands.
 * - Scalar overloads capture the scalar by value (inside `VectorScalarBinaryOperator`).
 *
 * @warning
 * Because expression nodes store operands by reference, avoid returning expression nodes
 * that reference temporaries from a function unless the evaluation happens before the
 * temporaries are destroyed.
 *
 * @see atlas::math::VectorExpression, atlas::math::VectorUnaryOperator,
 *      atlas::math::VectorBinaryOperator, atlas::math::VectorScalarBinaryOperator
 */

// ------------------------------------------------------------
// Alias helpers for common expression nodes
// ------------------------------------------------------------

/** @brief Unary negation node: computes `-x` lane-wise. */
template <typename T, typename E>
using VectorNeg = VectorUnaryOperator<T, E, detail::Negate<T>>;

/**
 * @brief Type-cast node: casts each lane to `To`.
 *
 * @tparam To Target scalar type.
 * @tparam FromExpr Source expression type.
 *
 * @note Uses `detail::TypeCast<From, To>` where `From = expr_value_t<FromExpr>`.
 */
template <typename To, typename FromExpr>
using VectorTypeCast = VectorUnaryOperator<To, FromExpr, detail::TypeCast<expr_value_t<FromExpr>, To>>;

/** @brief Addition node: computes `l + r` lane-wise. */
template <typename T, typename EL, typename ER>
using VectorAdd = VectorBinaryOperator<T, EL, ER, detail::Add<T>>;

/** @brief Subtraction node: computes `l - r` lane-wise. */
template <typename T, typename EL, typename ER>
using VectorSub = VectorBinaryOperator<T, EL, ER, detail::Sub<T>>;

/** @brief Multiplication node: computes `l * r` lane-wise. */
template <typename T, typename EL, typename ER>
using VectorMul = VectorBinaryOperator<T, EL, ER, detail::Mul<T>>;

/** @brief Division node: computes `l / r` lane-wise. */
template <typename T, typename EL, typename ER>
using VectorDiv = VectorBinaryOperator<T, EL, ER, detail::Div<T>>;

/** @brief Expression-scalar add: computes `e + s` lane-wise. */
template <typename T, typename E>
using VectorScalarAdd = VectorScalarBinaryOperator<T, E, detail::Add<T>>;

/** @brief Expression-scalar subtract: computes `e - s` lane-wise. */
template <typename T, typename E>
using VectorScalarSub = VectorScalarBinaryOperator<T, E, detail::Sub<T>>;

/** @brief Scalar-expression subtract: computes `s - e` lane-wise. */
template <typename T, typename E>
using VectorScalarRSub = VectorScalarBinaryOperator<T, E, detail::RSub<T>>;

/** @brief Expression-scalar multiply: computes `e * s` lane-wise. */
template <typename T, typename E>
using VectorScalarMul = VectorScalarBinaryOperator<T, E, detail::Mul<T>>;

/** @brief Expression-scalar divide: computes `e / s` lane-wise. */
template <typename T, typename E>
using VectorScalarDiv = VectorScalarBinaryOperator<T, E, detail::Div<T>>;

/** @brief Scalar-expression divide: computes `s / e` lane-wise. */
template <typename T, typename E>
using VectorScalarRDiv = VectorScalarBinaryOperator<T, E, detail::RDiv<T>>;

// ------------------------------------------------------------
// Comparison nodes (expression-expression)
// ------------------------------------------------------------

/**
 * @brief Less-than comparison node: `l < r` lane-wise.
 *
 * @note Returns a boolean expression (lane type = bool).
 *       Uses the left value type `expr_value_t<EL>` as the comparator's scalar type.
 */
template <typename EL, typename ER>
using VectorLess = VectorBinaryOperator<bool, EL, ER, detail::Less<expr_value_t<EL>>>;

/** @brief Less-equal comparison node: `l <= r` lane-wise. */
template <typename EL, typename ER>
using VectorLessEqual = VectorBinaryOperator<bool, EL, ER, detail::LessEqual<expr_value_t<EL>>>;

/** @brief Greater-than comparison node: `l > r` lane-wise. */
template <typename EL, typename ER>
using VectorGreater = VectorBinaryOperator<bool, EL, ER, detail::Greater<expr_value_t<EL>>>;

/** @brief Greater-equal comparison node: `l >= r` lane-wise. */
template <typename EL, typename ER>
using VectorGreaterEqual = VectorBinaryOperator<bool, EL, ER, detail::GreaterEqual<expr_value_t<EL>>>;

/** @brief Equality comparison node: `l == r` lane-wise. */
template <typename EL, typename ER>
using VectorEqual = VectorBinaryOperator<bool, EL, ER, detail::Equal<expr_value_t<EL>>>;

/** @brief Inequality comparison node: `l != r` lane-wise. */
template <typename EL, typename ER>
using VectorNotEqual = VectorBinaryOperator<bool, EL, ER, detail::NotEqual<expr_value_t<EL>>>;

// ------------------------------------------------------------
// Comparison nodes (expression-scalar and scalar-expression)
// ------------------------------------------------------------

/** @brief Expression-scalar less-than: `e < s` lane-wise. */
template <typename E>
using VectorScalarLess = VectorScalarBinaryOperator<bool, E, detail::Less<expr_value_t<E>>>;

/** @brief Scalar-expression less-than: `s < e` lane-wise. */
template <typename E>
using VectorScalarRLess = VectorScalarBinaryOperator<bool, E, detail::RLess<expr_value_t<E>>>;

/** @brief Expression-scalar less-equal: `e <= s` lane-wise. */
template <typename E>
using VectorScalarLessEqual = VectorScalarBinaryOperator<bool, E, detail::LessEqual<expr_value_t<E>>>;

/** @brief Scalar-expression less-equal: `s <= e` lane-wise. */
template <typename E>
using VectorScalarRLessEqual = VectorScalarBinaryOperator<bool, E, detail::RLessEqual<expr_value_t<E>>>;

/** @brief Expression-scalar greater-than: `e > s` lane-wise. */
template <typename E>
using VectorScalarGreater = VectorScalarBinaryOperator<bool, E, detail::Greater<expr_value_t<E>>>;

/** @brief Scalar-expression greater-than: `s > e` lane-wise. */
template <typename E>
using VectorScalarRGreater = VectorScalarBinaryOperator<bool, E, detail::RGreater<expr_value_t<E>>>;

/** @brief Expression-scalar greater-equal: `e >= s` lane-wise. */
template <typename E>
using VectorScalarGreaterEqual = VectorScalarBinaryOperator<bool, E, detail::GreaterEqual<expr_value_t<E>>>;

/** @brief Scalar-expression greater-equal: `s >= e` lane-wise. */
template <typename E>
using VectorScalarRGreaterEqual = VectorScalarBinaryOperator<bool, E, detail::RGreaterEqual<expr_value_t<E>>>;

/** @brief Expression-scalar equality: `e == s` lane-wise. */
template <typename E>
using VectorScalarEqual = VectorScalarBinaryOperator<bool, E, detail::Equal<expr_value_t<E>>>;

/** @brief Scalar-expression equality: `s == e` lane-wise. */
template <typename E>
using VectorScalarREqual = VectorScalarBinaryOperator<bool, E, detail::REqual<expr_value_t<E>>>;

/** @brief Expression-scalar inequality: `e != s` lane-wise. */
template <typename E>
using VectorScalarNotEqual = VectorScalarBinaryOperator<bool, E, detail::NotEqual<expr_value_t<E>>>;

/** @brief Scalar-expression inequality: `s != e` lane-wise. */
template <typename E>
using VectorScalarRNotEqual = VectorScalarBinaryOperator<bool, E, detail::RNotEqual<expr_value_t<E>>>;

// ------------------------------------------------------------
// Logical nodes (boolean expressions)
// ------------------------------------------------------------

/**
 * @brief Lane-wise logical AND node: `l && r` semantics implemented as a functor.
 *
 * @note Uses bitwise `&` operator overload in this header to build an expression node.
 *       The functor type is `detail::LogicalAnd<bool>`.
 */
template <typename EL, typename ER>
using VectorAnd = VectorBinaryOperator<bool, EL, ER, detail::LogicalAnd<bool>>;

/**
 * @brief Lane-wise logical OR node: `l || r` semantics implemented as a functor.
 *
 * @note Uses bitwise `|` operator overload in this header to build an expression node.
 *       The functor type is `detail::LogicalOr<bool>`.
 */
template <typename EL, typename ER>
using VectorOr = VectorBinaryOperator<bool, EL, ER, detail::LogicalOr<bool>>;

// ------------------------------------------------------------
// Arithmetic operators (expression-expression)
// ------------------------------------------------------------

/**
 * @brief Builds an expression node for lane-wise addition (`l + r`).
 *
 * @tparam EL Left expression type.
 * @tparam ER Right expression type.
 * @param l Left operand.
 * @param r Right operand.
 * @return Lazy expression node computing `l[i] + r[i]`.
 *
 * @details
 * The result lane type is the common type of the two operand lane types.
 */
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator+(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    // Unwrap CRTP (`()`) to pass the actual derived nodes into the expression constructor.
    return VectorAdd<T, EL, ER>(l(), r());
}

/**
 * @brief Builds an expression node for adding a scalar to an expression (`l + s`).
 *
 * @tparam T Lane/scalar type.
 * @tparam E Expression node type.
 * @param l Expression operand.
 * @param s Scalar value (captured by value inside the node).
 * @return Lazy expression node computing `l[i] + s`.
 */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator+(const VectorExpression<T, E>& l, const T& s) noexcept {
    // Stores scalar `s` by value in the expression node to avoid dangling references.
    return VectorScalarAdd<T, E>(l(), s);
}

/**
 * @brief Builds an expression node for adding an expression to a scalar (`s + r`).
 *
 * @tparam T Lane/scalar type.
 * @tparam E Expression node type.
 * @param s Scalar value.
 * @param r Expression operand.
 * @return Lazy expression node computing `r[i] + s`.
 */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator+(const T& s, const VectorExpression<T, E>& r) noexcept {
    // Commutative: reuse VectorScalarAdd with reversed parameter order.
    return VectorScalarAdd<T, E>(r(), s);
}

/**
 * @brief Builds an expression node for lane-wise subtraction (`l - r`).
 *
 * @tparam EL Left expression type.
 * @tparam ER Right expression type.
 * @param l Left operand.
 * @param r Right operand.
 * @return Lazy expression node computing `l[i] - r[i]`.
 */
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return VectorSub<T, EL, ER>(l(), r());
}

/**
 * @brief Builds an expression node for subtracting a scalar from an expression (`l - s`).
 *
 * @tparam T Lane/scalar type.
 * @tparam E Expression node type.
 * @param l Expression operand.
 * @param s Scalar value.
 * @return Lazy expression node computing `l[i] - s`.
 */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarSub<T, E>(l(), s);
}

/**
 * @brief Builds an expression node for subtracting an expression from a scalar (`s - r`).
 *
 * @tparam T Lane/scalar type.
 * @tparam E Expression node type.
 * @param s Scalar value.
 * @param r Expression operand.
 * @return Lazy expression node computing `s - r[i]`.
 *
 * @note Uses the dedicated reverse-sub functor (`detail::RSub<T>`).
 */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRSub<T, E>(r(), s);
}

/**
 * @brief Builds an expression node for lane-wise multiplication (`l * r`).
 *
 * @tparam EL Left expression type.
 * @tparam ER Right expression type.
 * @param l Left operand.
 * @param r Right operand.
 * @return Lazy expression node computing `l[i] * r[i]`.
 */
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator*(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return VectorMul<T, EL, ER>(l(), r());
}

/**
 * @brief Builds an expression node for multiplying an expression by a scalar (`l * s`).
 *
 * @tparam T Lane/scalar type.
 * @tparam E Expression node type.
 * @param l Expression operand.
 * @param s Scalar value.
 * @return Lazy expression node computing `l[i] * s`.
 */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator*(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarMul<T, E>(l(), s);
}

/**
 * @brief Builds an expression node for multiplying a scalar by an expression (`s * r`).
 *
 * @tparam T Lane/scalar type.
 * @tparam E Expression node type.
 * @param s Scalar value.
 * @param r Expression operand.
 * @return Lazy expression node computing `r[i] * s`.
 */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator*(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarMul<T, E>(r(), s);
}

/**
 * @brief Builds an expression node for lane-wise division (`l / r`).
 *
 * @tparam EL Left expression type.
 * @tparam ER Right expression type.
 * @param l Left operand.
 * @param r Right operand.
 * @return Lazy expression node computing `l[i] / r[i]`.
 */
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator/(const EL& l, const ER& r) noexcept {
    using T = std::common_type_t<expr_value_t<EL>, expr_value_t<ER>>;
    return VectorDiv<T, EL, ER>(l(), r());
}

/**
 * @brief Builds an expression node for dividing an expression by a scalar (`l / s`).
 *
 * @tparam T Lane/scalar type.
 * @tparam E Expression node type.
 * @param l Expression operand.
 * @param s Scalar value.
 * @return Lazy expression node computing `l[i] / s`.
 */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator/(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarDiv<T, E>(l(), s);
}

/**
 * @brief Builds an expression node for dividing a scalar by an expression (`s / r`).
 *
 * @tparam T Lane/scalar type.
 * @tparam E Expression node type.
 * @param s Scalar numerator.
 * @param r Expression denominator.
 * @return Lazy expression node computing `s / r[i]`.
 *
 * @note Uses the dedicated reverse-div functor (`detail::RDiv<T>`).
 */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator/(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRDiv<T, E>(r(), s);
}

// ------------------------------------------------------------
// Unary arithmetic
// ------------------------------------------------------------

/**
 * @brief Builds an expression node for unary negation (`-e`).
 *
 * @tparam E Expression type.
 * @param e Operand expression.
 * @return Lazy expression node computing `-e[i]`.
 */
template <VectorExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator-(const E& e) noexcept {
    using T = expr_value_t<E>;
    return VectorNeg<T, E>(e());
}

// ------------------------------------------------------------
// Comparisons (expression-expression)
// ------------------------------------------------------------

/**
 * @brief Builds a boolean mask expression for lane-wise less-than (`l < r`).
 *
 * @tparam EL Left expression type.
 * @tparam ER Right expression type.
 * @param l Left operand.
 * @param r Right operand.
 * @return Mask expression where lane `i` is `l[i] < r[i]`.
 */
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator<(const EL& l, const ER& r) noexcept {
    return VectorLess<EL, ER>(l(), r());
}

/** @brief Lane-wise less-equal (`l <= r`). */
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator<=(const EL& l, const ER& r) noexcept {
    return VectorLessEqual<EL, ER>(l(), r());
}

/** @brief Lane-wise greater-than (`l > r`). */
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator>(const EL& l, const ER& r) noexcept {
    return VectorGreater<EL, ER>(l(), r());
}

/** @brief Lane-wise greater-equal (`l >= r`). */
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator>=(const EL& l, const ER& r) noexcept {
    return VectorGreaterEqual<EL, ER>(l(), r());
}

/** @brief Lane-wise equality (`l == r`). */
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator==(const EL& l, const ER& r) noexcept {
    return VectorEqual<EL, ER>(l(), r());
}

/** @brief Lane-wise inequality (`l != r`). */
template <VectorExpressionType EL, VectorExpressionType ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator!=(const EL& l, const ER& r) noexcept {
    return VectorNotEqual<EL, ER>(l(), r());
}

// ------------------------------------------------------------
// Comparisons (expression-scalar and scalar-expression)
// ------------------------------------------------------------

/**
 * @brief Expression-scalar less-than (`l < s`).
 *
 * @tparam T Lane/scalar type.
 * @tparam E Expression type.
 * @param l Expression operand.
 * @param s Scalar operand.
 * @return Mask expression where lane `i` is `l[i] < s`.
 */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator<(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarLess<E>(l(), s);
}

/**
 * @brief Scalar-expression less-than (`s < r`).
 *
 * @tparam T Lane/scalar type.
 * @tparam E Expression type.
 * @param s Scalar operand.
 * @param r Expression operand.
 * @return Mask expression where lane `i` is `s < r[i]`.
 */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator<(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRLess<E>(r(), s);
}

/** @brief Expression-scalar less-equal (`l <= s`). */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator<=(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarLessEqual<E>(l(), s);
}

/** @brief Scalar-expression less-equal (`s <= r`). */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator<=(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRLessEqual<E>(r(), s);
}

/** @brief Expression-scalar greater-than (`l > s`). */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator>(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarGreater<E>(l(), s);
}

/** @brief Scalar-expression greater-than (`s > r`). */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator>(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRGreater<E>(r(), s);
}

/** @brief Expression-scalar greater-equal (`l >= s`). */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator>=(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarGreaterEqual<E>(l(), s);
}

/** @brief Scalar-expression greater-equal (`s >= r`). */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator>=(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRGreaterEqual<E>(r(), s);
}

/** @brief Expression-scalar equality (`l == s`). */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator==(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarEqual<E>(l(), s);
}

/** @brief Scalar-expression equality (`s == r`). */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator==(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarREqual<E>(r(), s);
}

/** @brief Expression-scalar inequality (`l != s`). */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator!=(const VectorExpression<T, E>& l, const T& s) noexcept {
    return VectorScalarNotEqual<E>(l(), s);
}

/** @brief Scalar-expression inequality (`s != r`). */
template <typename T, typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator!=(const T& s, const VectorExpression<T, E>& r) noexcept {
    return VectorScalarRNotEqual<E>(r(), s);
}

// ------------------------------------------------------------
// Boolean mask composition
// ------------------------------------------------------------

/**
 * @brief Builds a boolean mask expression for lane-wise logical AND (`l & r`).
 *
 * @tparam EL Left mask expression type.
 * @tparam ER Right mask expression type.
 * @param l Left mask expression (bool lanes).
 * @param r Right mask expression (bool lanes).
 * @return Mask expression where lane `i` is `l[i] && r[i]` (via `detail::LogicalAnd`).
 *
 * @note
 * This overload uses the bitwise operator `&` for mask composition to avoid C++'s short-circuiting
 * semantics (which do not apply to vector lanes). The resulting node evaluates lane-wise.
 */
template <typename EL, typename ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator&(const VectorExpression<bool, EL>& l, const VectorExpression<bool, ER>& r) noexcept {
    // Unwrap CRTP to avoid slicing and to store references to the derived nodes.
    return VectorAnd<EL, ER>(l(), r());
}

/**
 * @brief Builds a boolean mask expression for lane-wise logical OR (`l | r`).
 *
 * @tparam EL Left mask expression type.
 * @tparam ER Right mask expression type.
 * @param l Left mask expression (bool lanes).
 * @param r Right mask expression (bool lanes).
 * @return Mask expression where lane `i` is `l[i] || r[i]` (via `detail::LogicalOr`).
 *
 * @note
 * This overload uses the bitwise operator `|` for mask composition for the same reason as `operator&`.
 */
template <typename EL, typename ER>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
operator|(const VectorExpression<bool, EL>& l, const VectorExpression<bool, ER>& r) noexcept {
    return VectorOr<EL, ER>(l(), r());
}

} // namespace atlas::math
