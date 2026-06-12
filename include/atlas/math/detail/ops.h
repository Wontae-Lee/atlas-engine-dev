#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

namespace atlas::detail {

/**
 * @file ops.h
 * @brief Small stateless operator functors used by vector/matrix expression templates.
 *
 * @details
 * This header defines:
 * - A lightweight trait `expr_value_t<E>` to deduce the scalar element type of an expression-like `E`
 *   via `E::operator[](std::size_t)`.
 * - A collection of tiny function objects (functors) for arithmetic, comparisons, unary transforms,
 *   clamping, and boolean logic. These functors are designed to be inlined and usable in device code.
 *
 * Expression nodes (e.g., `VectorUnaryOperator`, `MatrixBinaryOperator`) store these functors by value
 * and call them for each element when evaluating an expression.
 *
 * Design goals:
 * - **Header-only** and trivially inlinable.
 * - **Device-friendly**: all call operators are annotated with `ATLAS_ALL_DEVICE`.
 * - **No hidden state** (except small parameter captures like `ClampScalar<T>::lo/hi`).
 *
 * @note
 * - Some functors implement "reversed" operands (e.g., `RSub`, `RDiv`, `RLess`) to support
 *   `scalar op expr` overloads without additional branching.
 * - `Abs` here is a simple branch-based implementation and does not handle NaNs specially.
 */

// ------------------------------------------------------------
// Expression value type deduction
// ------------------------------------------------------------

/**
 * @brief Deduces the element type of an expression-like type `E`.
 *
 * @tparam E An expression-like type that provides `operator[](std::size_t) const`.
 *
 * @details
 * The type is deduced by probing `E::operator[]` at a dummy index and taking the decayed type:
 * `std::decay_t<decltype(e[0])>`.
 *
 * This is used throughout the expression system to infer the scalar value type of an expression node.
 *
 * @note
 * This trait assumes that `E::operator[]` is valid for `std::size_t{0}` in an unevaluated context.
 */
template <typename E>
using expr_value_t = std::decay_t<decltype(std::declval<const E&>()[std::size_t { 0 }])>;

// ------------------------------------------------------------
// Arithmetic functors
// ------------------------------------------------------------

/**
 * @brief Element-wise addition functor.
 *
 * @tparam T Scalar type.
 */
template <typename T>
struct Add {
    /// @brief Returns `a + b`.
    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return a + b; }
};

/**
 * @brief Element-wise subtraction functor.
 *
 * @tparam T Scalar type.
 */
template <typename T>
struct Sub {
    /// @brief Returns `a - b`.
    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return a - b; }
};

/**
 * @brief Reversed subtraction functor for `scalar - expr` patterns.
 *
 * @tparam T Scalar type.
 *
 * @details
 * When an expression node stores operands as `(expr_value, scalar)`,
 * this functor can implement `scalar - expr_value` by returning `b - a`.
 */
template <typename T>
struct RSub {
    /// @brief Returns `b - a` (reversed operand order).
    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return b - a; }
};

/**
 * @brief Element-wise multiplication functor.
 *
 * @tparam T Scalar type.
 */
template <typename T>
struct Mul {
    /// @brief Returns `a * b`.
    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return a * b; }
};

/**
 * @brief Element-wise division functor.
 *
 * @tparam T Scalar type.
 */
template <typename T>
struct Div {
    /// @brief Returns `a / b`.
    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return a / b; }
};

/**
 * @brief Reversed division functor for `scalar / expr` patterns.
 *
 * @tparam T Scalar type.
 *
 * @details
 * When an expression node stores operands as `(expr_value, scalar)`,
 * this functor can implement `scalar / expr_value` by returning `b / a`.
 */
template <typename T>
struct RDiv {
    /// @brief Returns `b / a` (reversed operand order).
    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return b / a; }
};

// ------------------------------------------------------------
// Comparison functors (return bool)
// ------------------------------------------------------------

/** @brief Returns `a < b`. */
template <typename T>
struct Less {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return a < b; }
};

/** @brief Returns `a <= b`. */
template <typename T>
struct LessEqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return a <= b; }
};

/** @brief Returns `a > b`. */
template <typename T>
struct Greater {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return a > b; }
};

/** @brief Returns `a >= b`. */
template <typename T>
struct GreaterEqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return a >= b; }
};

/** @brief Returns `a == b`. */
template <typename T>
struct Equal {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return a == b; }
};

/** @brief Returns `a != b`. */
template <typename T>
struct NotEqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return a != b; }
};

// ------------------------------------------------------------
// Unary transform functors
// ------------------------------------------------------------

/**
 * @brief Unary negation functor.
 *
 * @tparam T Scalar type.
 */
template <typename T>
struct Negate {
    /// @brief Returns `-v`.
    ATLAS_ALL_DEVICE T
    operator()(T v) const noexcept { return -v; }
};

/**
 * @brief Absolute value functor.
 *
 * @tparam T Scalar type.
 *
 * @details
 * Implements `abs(v)` as `(v < 0) ? -v : v`.
 *
 * @note
 * - This is a simple branch-based implementation.
 * - For floating-point types, NaN behavior follows comparison semantics (NaN compares false).
 */
template <typename T>
struct Abs {
    /// @brief Returns `|v|` using a branch.
    ATLAS_ALL_DEVICE T
    operator()(T v) const noexcept { return (v < T(0)) ? -v : v; }
};

/**
 * @brief Sign function functor.
 *
 * @tparam T Scalar type.
 *
 * @details
 * Returns:
 * - `+1` if `v > 0`
 * - ` 0` if `v == 0`
 * - `-1` if `v < 0`
 *
 * Implemented as `(0 < v) - (v < 0)` which yields `1, 0, -1`.
 */
template <typename T>
struct Sign {
    /// @brief Returns the sign of `v` in {-1, 0, +1}.
    ATLAS_ALL_DEVICE T
    operator()(T v) const noexcept { return (T(0) < v) - (v < T(0)); }
};

/**
 * @brief Type-cast functor for lane-wise scalar conversion.
 *
 * @tparam From Source type.
 * @tparam To Target type.
 */
template <typename From, typename To>
struct TypeCast {
    /// @brief Returns `static_cast<To>(v)`.
    ATLAS_ALL_DEVICE To
    operator()(const From& v) const noexcept { return static_cast<To>(v); }
};

// ------------------------------------------------------------
// Min/max selection functors
// ------------------------------------------------------------

/**
 * @brief Returns the smaller of two values.
 *
 * @tparam T Scalar type.
 *
 * @note
 * This is a branch-based selection: `(b < a) ? b : a`.
 */
template <typename T>
struct CompareMin {
    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return (b < a) ? b : a; }
};

/**
 * @brief Returns the larger of two values.
 *
 * @tparam T Scalar type.
 *
 * @note
 * This is a branch-based selection: `(a < b) ? b : a`.
 */
template <typename T>
struct CompareMax {
    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return (a < b) ? b : a; }
};

// ------------------------------------------------------------
// Clamp
// ------------------------------------------------------------

/**
 * @brief Clamps a value into the inclusive range [`lo`, `hi`].
 *
 * @tparam T Scalar type.
 *
 * @details
 * Stores bounds as members:
 * - `lo`: lower bound
 * - `hi`: upper bound
 *
 * The call operator returns:
 * - `lo` if `v < lo`
 * - `hi` if `hi < v`
 * - otherwise `v`
 *
 * @note
 * No validation that `lo <= hi` is performed.
 */
template <typename T>
struct ClampScalar {
    T lo, hi;

    /// @brief Returns `v` clamped into [`lo`, `hi`].
    ATLAS_ALL_DEVICE T
    operator()(T v) const noexcept { return (v < lo) ? lo : ((hi < v) ? hi : v); }
};

// ------------------------------------------------------------
// Logical ops (for boolean expression nodes)
// ------------------------------------------------------------

/**
 * @brief Logical AND functor for boolean expressions.
 *
 * @tparam T Dummy template parameter to fit generic operator wiring.
 *
 * @details
 * The `(void)sizeof(T);` line prevents "unused template parameter" warnings in some toolchains.
 * The actual operands are `bool` and the result is `bool`.
 */
template <typename T>
struct LogicalAnd {
    /// @brief Returns `l && r`.
    ATLAS_ALL_DEVICE bool
    operator()(bool l, bool r) const noexcept {
        (void)sizeof(T);
        return l && r;
    }
};

/**
 * @brief Logical OR functor for boolean expressions.
 *
 * @tparam T Dummy template parameter to fit generic operator wiring.
 */
template <typename T>
struct LogicalOr {
    /// @brief Returns `l || r`.
    ATLAS_ALL_DEVICE bool
    operator()(bool l, bool r) const noexcept {
        (void)sizeof(T);
        return l || r;
    }
};

// ------------------------------------------------------------
// Reversed comparisons (support scalar op expr overloads)
// ------------------------------------------------------------

/** @brief Returns `b < a` (reversed operand order). */
template <typename T>
struct RLess {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return b < a; }
};

/** @brief Returns `b <= a` (reversed operand order). */
template <typename T>
struct RLessEqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return b <= a; }
};

/** @brief Returns `b > a` (reversed operand order). */
template <typename T>
struct RGreater {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return b > a; }
};

/** @brief Returns `b >= a` (reversed operand order). */
template <typename T>
struct RGreaterEqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return b >= a; }
};

/** @brief Returns `b == a` (reversed operand order). */
template <typename T>
struct REqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return b == a; }
};

/** @brief Returns `b != a` (reversed operand order). */
template <typename T>
struct RNotEqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return b != a; }
};

} // namespace atlas::detail
