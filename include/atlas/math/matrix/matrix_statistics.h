#pragma once
#include <atlas/math/matrix/matrix_expression.h>
#include <atlas/math/matrix/matrix_reductions.h>
#include <cmath>
#include <cstddef>
#include <type_traits>
namespace atlas {

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    mean(const E& expr) noexcept {
    using T             = expr_value_t<E>;
    const std::size_t n = expr().size();
    if (n == 0) return T(0);

    return sum(expr) / static_cast<T>(n);
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    variance_population(const E& expr) noexcept {
    using T             = expr_value_t<E>;
    const auto& a       = expr();
    const std::size_t n = a.size();
    if (n == 0) return T(0);
    const T m = mean(expr);
    double M2 = 0.0;
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        const auto d = static_cast<double>(a[i] - m);
        M2 += d * d;
    }
    return static_cast<T>(M2 / static_cast<double>(n));
}

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

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    stddev_population(const E& expr) noexcept {
    using T = expr_value_t<E>;
    return static_cast<T>(std::sqrt(static_cast<double>(variance_population(expr))));
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    stddev_sample(const E& expr) noexcept {
    using T = expr_value_t<E>;
    return static_cast<T>(std::sqrt(static_cast<double>(variance_sample(expr))));
}

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

template <MatrixExpressionType EX, MatrixExpressionType EY>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
correlation(const EX& x, const EY& y) noexcept {
    using T    = std::common_type_t<expr_value_t<EX>, expr_value_t<EY>>;
    const T sx = stddev_sample(x);
    const T sy = stddev_sample(y);
    if (sx == T(0) || sy == T(0)) return T(0);
    return covariance_sample(x, y) / (sx * sy);
}

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
        M3 += d * d * d;
    }
    if (M2 == 0.0) return T(0);
    const double mu2 = M2 / static_cast<double>(n);
    const double mu3 = M3 / static_cast<double>(n);
    return static_cast<T>(mu3 / std::pow(mu2, 1.5));
}

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
        M2 += d2sq;
        M4 += d2sq * d2sq;
    }
    if (M2 == 0.0) return T(0);
    const double mu2 = M2 / static_cast<double>(n);
    const double mu4 = M4 / static_cast<double>(n);
    return static_cast<T>(mu4 / (mu2 * mu2) - 3.0);
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    expr_value_t<E>
    logsumexp(const E& expr) noexcept {
    using T             = expr_value_t<E>;
    const auto& v       = expr();
    const std::size_t n = v.size();
    if (n == 0) return T(0);
    T mx = v[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < n; ++i) {
        if (mx < v[i]) mx = v[i];
    }
    double acc = 0.0;
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        acc += std::exp(static_cast<double>(v[i] - mx));
    }
    return static_cast<T>(std::log(acc) + static_cast<double>(mx));
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
softmax(const E& expr) noexcept {
    using T     = expr_value_t<E>;
    const T lse = logsumexp(expr);
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

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
zscore(const E& expr) noexcept {
    using T   = expr_value_t<E>;
    const T m = mean(expr);
    const T s = stddev_sample(expr);
    if (s == T(0)) {

        struct ZeroOp {
            T mid;
            ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
                T
                operator()(T) const noexcept { return T(0); }
        };
        return MatrixUnaryOperator<T, E, ZeroOp>(expr(), ZeroOp { m });
    }

    struct ZOp {
        T m;
        T invs;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
            T
            operator()(T x) const noexcept { return (x - m) * invs; }
    };
    return MatrixUnaryOperator<T, E, ZOp>(expr(), ZOp { m, T(1) / s });
}

template <MatrixExpressionType E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
normalize_minmax(const E& expr, expr_value_t<E> a, expr_value_t<E> b) noexcept {
    using T       = expr_value_t<E>;
    const T vmin  = min(expr);
    const T vmax  = max(expr);
    const T range = vmax - vmin;
    if (range == T(0)) {

        struct MidOp {
            T mid;
            ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
                T
                operator()(T) const noexcept { return mid; }
        };
        const T mid = (a + b) * T(0.5);
        return MatrixUnaryOperator<T, E, MidOp>(expr(), MidOp { mid });
    }

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

}