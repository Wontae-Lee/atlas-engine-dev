#pragma once
#include <atlas/math/matrix/matrix_expression.h>
#include <atlas/math/matrix/matrix_reductions.h>
#include <cmath>
#include <cstddef>
#include <type_traits>
namespace atlas::math {
/**
 * @file matrix_statistics.h
 * @brief Statistical utilities for matrix expressions (mean/variance/covariance, softmax, z-score, min-max scaling).
 *
 * @details
 * This header provides common statistical operations on Atlas matrix expression templates.
 * Functions fall into two categories:
 *
 * 1) **Scalar reductions (eager)**:
 *    - `mean`
 *    - `variance_population`, `variance_sample`
 *    - `stddev_population`, `stddev_sample`
 *    - `covariance_population`, `covariance_sample`, `correlation`
 *    - `skewness`, `kurtosis_excess`
 *    - `logsumexp`
 *
 * 2) **Element-wise transforms (lazy)**:
 *    - `softmax`
 *    - `zscore`
 *    - `normalize_minmax`
 *
 * Most routines iterate over the expression's **linear indexing domain** `[0, size())`.
 * The mapping from linear index to (row, col) is defined by the concrete matrix type.
 *
 * Numerical behavior / policies:
 * - Accumulators use `double` to improve precision and reduce rounding error.
 * - Empty inputs return 0 for scalar reductions and a safe output for transforms where applicable.
 * - Degenerate cases (e.g., zero standard deviation, zero range) return zeros or midpoint scaling
 *   to avoid division-by-zero and NaNs.
 *
 * @note
 * - Paired-input functions (`covariance_*`, `correlation`) assume `x.size() == y.size()`.
 * - `normalize_minmax` depends on `min(expr)` / `max(expr)` from `matrix/reductions.h`.
 *
 * @see atlas::math::sum, atlas::math::min, atlas::math::max
 */
// ------------------------------------------------------------
// Mean
// ------------------------------------------------------------
/**
 * @brief Computes the arithmetic mean of all elements in a matrix expression.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return Mean value \f$\mu = \frac{1}{n}\sum_i x_i\f$.
 *
 * @details
 * Uses `sum(expr)` from `matrix/reductions.h` and divides by `n = size()`.
 * If `n == 0`, returns `T(0)`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    mean(const E& expr) noexcept {
    using T             = expr_value_t<E>;
    const std::size_t n = expr().size(); // Total element count.
    if (n == 0) return T(0);             // Empty input -> 0 by policy.
    // Sum is eager; division is in element type T.
    return sum(expr) / static_cast<T>(n);
}
// ------------------------------------------------------------
// Variance / standard deviation
// ------------------------------------------------------------
/**
 * @brief Computes the population variance of all elements.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return Population variance \f$\sigma^2 = \frac{1}{n}\sum_i (x_i-\mu)^2\f$.
 *
 * @details
 * Two-pass approach:
 * 1) Compute mean \f$\mu\f$
 * 2) Accumulate squared deviations in `double` for accuracy
 *
 * Returns 0 if `n == 0`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    variance_population(const E& expr) noexcept {
    using T             = expr_value_t<E>;
    const auto& a       = expr();   // Underlying derived expression.
    const std::size_t n = a.size(); // Number of elements.
    if (n == 0) return T(0);
    const T m = mean(expr); // Mean (eager reduction).
    double M2 = 0.0;        // Sum of squared deviations.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        const auto d = static_cast<double>(a[i] - m);
        M2 += d * d;
    }
    return static_cast<T>(M2 / static_cast<double>(n));
}
/**
 * @brief Computes the sample variance of all elements (Bessel's correction).
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return Sample variance \f$s^2 = \frac{1}{n-1}\sum_i (x_i-\bar{x})^2\f$.
 *
 * @details
 * Returns 0 if `n <= 1` to avoid division-by-zero.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    variance_sample(const E& expr) noexcept {
    using T             = expr_value_t<E>;
    const auto& a       = expr();
    const std::size_t n = a.size();
    if (n <= 1) return T(0);
    const T m = mean(expr);
    double M2 = 0.0;
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        const auto d = static_cast<double>(a[i] - m);
        M2 += d * d;
    }
    return static_cast<T>(M2 / static_cast<double>(n - 1));
}
/**
 * @brief Computes the population standard deviation.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return \f$\sigma = \sqrt{\sigma^2}\f$.
 *
 * @note
 * Square root is computed in `double` and cast back to `T`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    stddev_population(const E& expr) noexcept {
    using T = expr_value_t<E>;
    return static_cast<T>(std::sqrt(static_cast<double>(variance_population(expr))));
}
/**
 * @brief Computes the sample standard deviation.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return \f$s = \sqrt{s^2}\f$.
 *
 * @note
 * Square root is computed in `double` and cast back to `T`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    stddev_sample(const E& expr) noexcept {
    using T = expr_value_t<E>;
    return static_cast<T>(std::sqrt(static_cast<double>(variance_sample(expr))));
}
// ------------------------------------------------------------
// Covariance / correlation
// ------------------------------------------------------------
/**
 * @brief Computes the population covariance between two matrix expressions (flattened).
 *
 * @tparam EX Expression type for x.
 * @tparam EY Expression type for y.
 * @param x First input.
 * @param y Second input.
 * @return \f$\mathrm{cov}(x,y)=\frac{1}{n}\sum_i (x_i-\mu_x)(y_i-\mu_y)\f$.
 *
 * @details
 * Returns 0 if `n == 0`.
 *
 * @warning
 * Assumes `x.size() == y.size()`; no checks are performed.
 */
template <MatrixExpressionType EX, MatrixExpressionType EY>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
covariance_population(const EX& x, const EY& y) noexcept {
    using T             = std::common_type_t<expr_value_t<EX>, expr_value_t<EY>>;
    const auto& a       = x();
    const auto& b       = y();
    const std::size_t n = a.size();
    if (n == 0) return T(0);
    const T mx = mean(x);
    const T my = mean(y);
    double acc = 0.0;
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        acc += static_cast<double>((a[i] - mx) * (b[i] - my));
    }
    return static_cast<T>(acc / static_cast<double>(n));
}
/**
 * @brief Computes the sample covariance between two matrix expressions (flattened).
 *
 * @tparam EX Expression type for x.
 * @tparam EY Expression type for y.
 * @param x First input.
 * @param y Second input.
 * @return \f$\mathrm{cov}_s(x,y)=\frac{1}{n-1}\sum_i (x_i-\bar{x})(y_i-\bar{y})\f$.
 *
 * @details
 * Returns 0 if `n <= 1`.
 *
 * @warning
 * Assumes `x.size() == y.size()`; no checks are performed.
 */
template <MatrixExpressionType EX, MatrixExpressionType EY>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
covariance_sample(const EX& x, const EY& y) noexcept {
    using T             = std::common_type_t<expr_value_t<EX>, expr_value_t<EY>>;
    const auto& a       = x();
    const auto& b       = y();
    const std::size_t n = a.size();
    if (n <= 1) return T(0);
    const T mx = mean(x);
    const T my = mean(y);
    double acc = 0.0;
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        acc += static_cast<double>((a[i] - mx) * (b[i] - my));
    }
    return static_cast<T>(acc / static_cast<double>(n - 1));
}
/**
 * @brief Computes the sample Pearson correlation coefficient between two expressions.
 *
 * @tparam EX Expression type for x.
 * @tparam EY Expression type for y.
 * @param x First input.
 * @param y Second input.
 * @return \f$\rho = \frac{\mathrm{cov}_s(x,y)}{s_x s_y}\f$.
 *
 * @details
 * Uses sample standard deviations and sample covariance.
 * Returns 0 if either standard deviation is zero to avoid division-by-zero.
 */
template <MatrixExpressionType EX, MatrixExpressionType EY>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
correlation(const EX& x, const EY& y) noexcept {
    using T    = std::common_type_t<expr_value_t<EX>, expr_value_t<EY>>;
    const T sx = stddev_sample(x);
    const T sy = stddev_sample(y);
    if (sx == T(0) || sy == T(0)) return T(0); // Degenerate inputs -> 0 by policy.
    return covariance_sample(x, y) / (sx * sy);
}
// ------------------------------------------------------------
// Skewness / kurtosis
// ------------------------------------------------------------
/**
 * @brief Computes the (population) skewness of the flattened elements.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return \f$\gamma_1 = \mu_3 / \mu_2^{3/2}\f$ (moment skewness).
 *
 * @details
 * Computes:
 * - \f$\mu_2 = \frac{1}{n}\sum (x_i-\mu)^2\f$
 * - \f$\mu_3 = \frac{1}{n}\sum (x_i-\mu)^3\f$
 *
 * Returns 0 if `n == 0` or if variance is zero.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    skewness(const E& expr) noexcept {
    using T             = expr_value_t<E>;
    const auto& a       = expr();
    const std::size_t n = a.size();
    if (n == 0) return T(0);
    const T m = mean(expr);
    double M2 = 0.0;
    double M3 = 0.0;
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        const auto d = static_cast<double>(a[i] - m);
        M2 += d * d;
        M3 += d * d * d; // d^3
    }
    if (M2 == 0.0) return T(0); // Constant input -> 0 skewness by policy.
    const double mu2 = M2 / static_cast<double>(n);
    const double mu3 = M3 / static_cast<double>(n);
    return static_cast<T>(mu3 / std::pow(mu2, 1.5));
}
/**
 * @brief Computes the excess kurtosis of the flattened elements.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return \f$\gamma_2 = \mu_4 / \mu_2^2 - 3\f$ (excess kurtosis).
 *
 * @details
 * Computes:
 * - \f$\mu_2 = \frac{1}{n}\sum (x_i-\mu)^2\f$
 * - \f$\mu_4 = \frac{1}{n}\sum (x_i-\mu)^4\f$
 *
 * Returns 0 if `n == 0` or if variance is zero.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    kurtosis_excess(const E& expr) noexcept {
    using T             = expr_value_t<E>;
    const auto& a       = expr();
    const std::size_t n = a.size();
    if (n == 0) return T(0);
    const T m = mean(expr);
    double M2 = 0.0;
    double M4 = 0.0;
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        const auto d2     = static_cast<double>(a[i] - m);
        const double d2sq = d2 * d2;
        M2 += d2sq;        // d^2
        M4 += d2sq * d2sq; // d^4
    }
    if (M2 == 0.0) return T(0);
    const double mu2 = M2 / static_cast<double>(n);
    const double mu4 = M4 / static_cast<double>(n);
    return static_cast<T>(mu4 / (mu2 * mu2) - 3.0);
}
// ------------------------------------------------------------
// log-sum-exp / softmax
// ------------------------------------------------------------
/**
 * @brief Computes log-sum-exp in a numerically stable manner.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return \f$\log\sum_i \exp(x_i)\f$.
 *
 * @details
 * Uses stabilization:
 * \f[
 *   \log\sum_i \exp(x_i) = m + \log\sum_i \exp(x_i-m), \quad m=\max_i x_i.
 * \f]
 *
 * Returns 0 if `n == 0`.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    logsumexp(const E& expr) noexcept {
    using T             = expr_value_t<E>;
    const auto& v       = expr();
    const std::size_t n = v.size();
    if (n == 0) return T(0);
    T mx = v[0]; // Maximum element for stabilization.
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        if (mx < v[i]) mx = v[i];
    }
    double acc = 0.0; // Sum of exp(x_i - mx) in double.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        acc += std::exp(static_cast<double>(v[i] - mx));
    }
    return static_cast<T>(std::log(acc) + static_cast<double>(mx));
}
/**
 * @brief Returns a lazy expression computing the softmax of the input (flattened).
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return Expression node evaluating \f$\exp(x_i - \text{LSE}(x))\f$ per element.
 *
 * @details
 * Computes `lse = logsumexp(expr)` eagerly, then returns a unary node applying:
 * `exp(v - lse)` per element.
 *
 * @note
 * This produces the normalized softmax values because subtracting `logsumexp`
 * makes the exponentials sum to 1 (up to floating-point error).
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
softmax(const E& expr) noexcept {
    using T     = expr_value_t<E>;
    const T lse = logsumexp(expr); // Eager scalar reduction.
    struct SoftmaxOp {
        T lse;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
            T
            operator()(T v) const noexcept {
            return static_cast<T>(std::exp(static_cast<double>(v - lse)));
        }
    };
    return MatrixUnaryOperator<T, E, SoftmaxOp>(expr(), SoftmaxOp { lse });
}
// ------------------------------------------------------------
// Z-score / min-max normalization
// ------------------------------------------------------------
/**
 * @brief Returns a lazy expression computing z-score standardized values.
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @return Expression node evaluating \f$(x_i-\mu)/s\f$ per element.
 *
 * @details
 * Uses:
 * - `m = mean(expr)`
 * - `s = stddev_sample(expr)`
 *
 * Degenerate handling:
 * - If `s == 0`, returns an expression that produces zeros (policy choice).
 *
 * @note
 * The `ZeroOp` struct includes a member (`mid`) that is not used by `operator()`.
 * It is preserved here as written, but could be removed to avoid confusion.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
zscore(const E& expr) noexcept {
    using T   = expr_value_t<E>;
    const T m = mean(expr);
    const T s = stddev_sample(expr);
    if (s == T(0)) {
        // Constant input: define z-score as 0 everywhere by policy.
        struct ZeroOp {
            T mid; // Unused; kept to match current layout.
            ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
                T
                operator()(T) const noexcept { return T(0); }
        };
        return MatrixUnaryOperator<T, E, ZeroOp>(expr(), ZeroOp { m });
    }
    // Precompute reciprocal to avoid division per element.
    struct ZOp {
        T m;
        T invs;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
            T
            operator()(T x) const noexcept { return (x - m) * invs; }
    };
    return MatrixUnaryOperator<T, E, ZOp>(expr(), ZOp { m, T(1) / s });
}
/**
 * @brief Returns a lazy expression that min-max normalizes the input into [a, b].
 *
 * @tparam E Matrix expression type.
 * @param expr Input expression.
 * @param a Target lower bound.
 * @param b Target upper bound.
 * @return Expression mapping each element to `a + (x - vmin) * ((b - a) / (vmax - vmin))`,
 *         or to the midpoint `(a+b)/2` if the input range is zero.
 *
 * @details
 * Uses `vmin = min(expr)` and `vmax = max(expr)` (eager reductions).
 *
 * Degenerate handling:
 * - If `vmax - vmin == 0`, returns a constant midpoint expression.
 */
template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
normalize_minmax(const E& expr, expr_value_t<E> a, expr_value_t<E> b) noexcept {
    using T       = expr_value_t<E>;
    const T vmin  = min(expr); // Eager min reduction.
    const T vmax  = max(expr); // Eager max reduction.
    const T range = vmax - vmin;
    if (range == T(0)) {
        // Constant input: map everything to the midpoint of [a, b].
        struct MidOp {
            T mid;
            ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
                T
                operator()(T) const noexcept { return mid; }
        };
        const T mid = (a + b) * T(0.5);
        return MatrixUnaryOperator<T, E, MidOp>(expr(), MidOp { mid });
    }
    // Affine scaling: a + (x - vmin) * s, where s = (b - a) / range.
    struct ScaleOp {
        T vmin;
        T s;
        T a;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
            T
            operator()(T x) const noexcept { return a + (x - vmin) * s; }
    };
    return MatrixUnaryOperator<T, E, ScaleOp>(expr(), ScaleOp { vmin, (b - a) / range, a });
}
} // namespace atlas::math
