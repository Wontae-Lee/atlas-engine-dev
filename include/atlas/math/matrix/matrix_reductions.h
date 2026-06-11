#pragma once
#include <atlas/math/matrix/matrix_expression.h>
#include <cmath>
#include <cstddef>
#include <type_traits>
namespace atlas::math {
/**
 * @file matrix_reductions.h
 * @brief Scalar reductions for matrix expressions (sum, min/max, norms, dot, argmin/argmax).
 *
 * @details
 * This header defines **eager** reduction utilities that consume a matrix expression and
 * produce a scalar result (or index). Unlike element-wise operators, these functions
 * iterate the expression immediately and return a concrete value.
 *
 * The reductions operate over the expression's **linear indexing domain** `[0, size())`.
 * The mapping between linear index and (row, col) is defined by the underlying matrix type.
 *
 * Provided reductions:
 * - `sum`, `min`, `max`
 * - `length_squared`, `length` (L2 norm of the flattened matrix)
 * - `dot` (flattened dot product)
 * - `distance` (L2 distance between two flattened matrices)
 * - `argmin`, `argmax`
 * - `argabsmin`, `argabsmax` (indices of smallest/largest absolute value)
 *
 * Numerical behavior:
 * - `length`/`distance` use the scalar type's square-root overload and cast back.
 * - Most reductions assume non-empty input when they access `e[0]`.
 *
 * @note
 * - Many functions in this file read `e[0]` without checking `n == 0`. If empty matrices
 *   are possible, callers should guard before calling, or the implementation should add
 *   explicit `n == 0` handling consistently.
 * - Shape compatibility is assumed for paired-input reductions (`dot`, `distance`).
 *
 * @see MatrixExpression, MatrixExpressionType
 */
// ------------------------------------------------------------
// Sum
// ------------------------------------------------------------
/**
 * @brief Computes the sum of all elements in a matrix expression.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return Sum of all elements.
 *
 * @details
 * - If `size() == 0`, returns `T(0)`.
 * - Otherwise initializes the accumulator from `e[0]` to avoid a default-zero requirement
 *   on the element type (still returns early on empty input).
 *
 * @note
 * Accumulation is performed in the expression element type `expr_value_t<E>`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
sum(const E& expr) noexcept {
    const auto& e       = expr();          // Bind to derived expression node (no copy).
    const std::size_t n = e.size();        // Total element count.
    if (n == 0) return expr_value_t<E>(0); // Empty matrix: define sum as 0.
    expr_value_t<E> acc = e[0];            // Seed accumulator with first element.
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        acc += e[i]; // Add remaining elements.
    }
    return acc;
}
// ------------------------------------------------------------
// Min / Max
// ------------------------------------------------------------
/**
 * @brief Returns the minimum element of a matrix expression.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return Minimum element.
 *
 * @warning
 * This implementation assumes `size() > 0` and reads `e[0]` unconditionally.
 * Calling with an empty expression is undefined behavior.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
min(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    auto m              = e[0]; // Initialize min with the first element (requires n>0).
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        if (e[i] < m) m = e[i]; // Update min.
    }
    return m;
}
/**
 * @brief Returns the maximum element of a matrix expression.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return Maximum element.
 *
 * @warning
 * This implementation assumes `size() > 0` and reads `e[0]` unconditionally.
 * Calling with an empty expression is undefined behavior.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
max(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    auto m              = e[0]; // Initialize max with the first element (requires n>0).
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        if (m < e[i]) m = e[i]; // Update max.
    }
    return m;
}
// ------------------------------------------------------------
// Norms
// ------------------------------------------------------------
/**
 * @brief Computes the squared L2 norm of a matrix expression (flattened).
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return \f$\sum_i x_i^2\f$ where `x_i` iterates over linear indexing.
 *
 * @details
 * Accumulates in the element type `T = expr_value_t<E>`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
length_squared(const E& expr) noexcept {
    using T             = expr_value_t<E>;
    const auto& e       = expr();
    const std::size_t n = e.size();
    T acc               = T(0); // Start from zero.
    if constexpr (std::is_floating_point_v<T>) {
        using std::fma;
        ATLAS_UNROLL
        for (std::size_t i = 0; i < n; ++i) {
            const T v = e[i];
            acc = fma(v, v, acc);
        }
    } else {
        ATLAS_UNROLL
        for (std::size_t i = 0; i < n; ++i) {
            acc += e[i] * e[i]; // Sum of squares.
        }
    }
    return acc;
}
/**
 * @brief Computes the L2 norm of a matrix expression (flattened).
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return \f$\sqrt{\sum_i x_i^2}\f$.
 *
 * @note
 * The square root is computed with the scalar type's overload and cast back to `expr_value_t<E>`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE expr_value_t<E>
length(const E& expr) noexcept {
    using T = expr_value_t<E>;
    using std::sqrt;
    return static_cast<T>(sqrt(length_squared(expr)));
}
// ------------------------------------------------------------
// Dot / Distance
// ------------------------------------------------------------
/**
 * @brief Computes the flattened dot product of two matrix expressions.
 *
 * @tparam EA First matrix expression type.
 * @tparam EB Second matrix expression type.
 * @param a First input.
 * @param b Second input.
 * @return \f$\sum_i a_i b_i\f$ over linear indexing.
 *
 * @details
 * Result type is `std::common_type_t<expr_value_t<EA>, expr_value_t<EB>>`.
 *
 * @warning
 * Assumes both expressions have the same `size()`. No checks are performed.
 */
template <MatrixExpressionType EA, MatrixExpressionType EB>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
dot(const EA& a, const EB& b) noexcept {
    using T             = std::common_type_t<expr_value_t<EA>, expr_value_t<EB>>;
    const auto& x       = a();      // Bind derived node for a.
    const auto& y       = b();      // Bind derived node for b.
    const std::size_t n = x.size(); // Iterate using size from a.
    T acc               = T(0);
    if constexpr (std::is_floating_point_v<T>) {
        using std::fma;
        ATLAS_UNROLL
        for (std::size_t i = 0; i < n; ++i) {
            acc = fma(static_cast<T>(x[i]), static_cast<T>(y[i]), acc);
        }
    } else {
        ATLAS_UNROLL
        for (std::size_t i = 0; i < n; ++i) {
            acc += x[i] * y[i]; // Multiply element-wise and accumulate.
        }
    }
    return acc;
}
/**
 * @brief Computes the L2 distance between two matrix expressions (flattened).
 *
 * @tparam EA First matrix expression type.
 * @tparam EB Second matrix expression type.
 * @param a First input.
 * @param b Second input.
 * @return \f$\sqrt{\sum_i (a_i - b_i)^2}\f$.
 *
 * @details
 * Forms a lazy difference expression `diff = a - b`, then reduces it via `length_squared`.
 * The square root is computed with the common scalar type's overload.
 *
 * @warning
 * Assumes both expressions have the same shape/size. No checks are performed.
 */
template <MatrixExpressionType EA, MatrixExpressionType EB>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
distance(const EA& a, const EB& b) noexcept {
    using T         = std::common_type_t<expr_value_t<EA>, expr_value_t<EB>>;
    const auto diff = a - b; // Lazy element-wise difference.
    using std::sqrt;
    return static_cast<T>(sqrt(length_squared(diff)));
}
// ------------------------------------------------------------
// Argmin / Argmax
// ------------------------------------------------------------
/**
 * @brief Returns the linear index of the minimum element.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return Index `i` such that `expr[i]` is minimal.
 *
 * @warning
 * Assumes `size() > 0` and reads `e[0]` unconditionally.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
argmin(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    std::size_t idx     = 0;
    auto m              = e[0]; // Current minimum value (requires n>0).
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        if (e[i] < m) {
            m   = e[i];
            idx = i; // Track index of best value.
        }
    }
    return idx;
}
/**
 * @brief Returns the linear index of the maximum element.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return Index `i` such that `expr[i]` is maximal.
 *
 * @warning
 * Assumes `size() > 0` and reads `e[0]` unconditionally.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
argmax(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    std::size_t idx     = 0;
    auto m              = e[0]; // Current maximum value (requires n>0).
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        if (m < e[i]) {
            m   = e[i];
            idx = i; // Track index of best value.
        }
    }
    return idx;
}
// ------------------------------------------------------------
// Argabsmin / Argabsmax
// ------------------------------------------------------------
/**
 * @brief Returns the linear index of the element with minimum absolute value.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return Index `i` minimizing `abs(expr[i])`.
 *
 * @warning
 * Assumes `size() > 0` and reads `e[0]` unconditionally.
 *
 * @note
 * Uses `std::abs` on `e[i]`. Ensure `std::abs` is well-defined for the element type
 * (typically arithmetic types).
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
argabsmin(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    std::size_t idx     = 0;
    auto mm             = std::abs(e[0]); // Current best absolute magnitude (requires n>0).
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        const auto v = std::abs(e[i]); // Absolute value of current element.
        if (v < mm) {
            mm  = v;
            idx = i;
        }
    }
    return idx;
}
/**
 * @brief Returns the linear index of the element with maximum absolute value.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return Index `i` maximizing `abs(expr[i])`.
 *
 * @warning
 * Assumes `size() > 0` and reads `e[0]` unconditionally.
 *
 * @note
 * Uses `std::abs` on `e[i]`. Ensure `std::abs` is well-defined for the element type
 * (typically arithmetic types).
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
argabsmax(const E& expr) noexcept {
    const auto& e       = expr();
    const std::size_t n = e.size();
    std::size_t idx     = 0;
    auto mm             = std::abs(e[0]); // Current best absolute magnitude (requires n>0).
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        const auto v = std::abs(e[i]);
        if (mm < v) {
            mm  = v;
            idx = i;
        }
    }
    return idx;
}
} // namespace atlas::math
