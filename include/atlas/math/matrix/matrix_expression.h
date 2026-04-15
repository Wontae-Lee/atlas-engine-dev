#pragma once
#include <atlas/math/detail/ops.h>
#include <cstddef>
#include <type_traits>
namespace atlas::math {

template <typename T>
class MatrixExpressionBase { };

template <typename T, typename E>
class MatrixExpression : public MatrixExpressionBase<T> {
public:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    rows() const noexcept {

        return static_cast<const E&>(*this).rows();
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    cols() const noexcept {

        return static_cast<const E&>(*this).cols();
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    size() const noexcept {

        return rows() * cols();
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {

        return static_cast<const E&>(*this)[i];
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(std::size_t r, std::size_t c) const noexcept {

        return static_cast<const E&>(*this)(r, c);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const E&
    operator()() const noexcept {

        return static_cast<const E&>(*this);
    }
};

template <typename E>
using expr_value_t = detail::expr_value_t<E>;

template <typename E>
concept MatrixExpressionType = std::is_base_of_v<MatrixExpression<expr_value_t<E>, E>, E>;

template <typename T, typename E, typename Operator>
class MatrixUnaryOperator
    : public MatrixExpression<T, MatrixUnaryOperator<T, E, Operator>> {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit MatrixUnaryOperator(const E& operand, Operator op = Operator {}) noexcept
        : _e(operand)
        , _op(op) { }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    rows() const noexcept { return _e.rows(); }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    cols() const noexcept { return _e.cols(); }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {

        return _op(_e[i]);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(std::size_t r, std::size_t c) const noexcept {

        return _op(_e(r, c));
    }

private:
    const E& _e;
    Operator _op;
};

template <typename T, typename EL, typename ER, typename Operator>
class MatrixBinaryOperator
    : public MatrixExpression<T, MatrixBinaryOperator<T, EL, ER, Operator>> {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    MatrixBinaryOperator(const EL& l, const ER& r, Operator op = Operator {}) noexcept
        : _l(l)
        , _r(r)
        , _op(op) { }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    rows() const noexcept { return _l.rows(); }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    cols() const noexcept { return _l.cols(); }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {

        return _op(_l[i], _r[i]);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(std::size_t r, std::size_t c) const noexcept {

        return _op(_l(r, c), _r(r, c));
    }

private:
    const EL& _l;
    const ER& _r;
    Operator _op;
};

template <typename T, typename E, typename Operator>
class MatrixScalarRight
    : public MatrixExpression<T, MatrixScalarRight<T, E, Operator>> {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    MatrixScalarRight(const E& e, T v, Operator op = Operator {}) noexcept
        : _e(e)
        , _v(v)
        , _op(op) { }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    rows() const noexcept { return _e.rows(); }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    cols() const noexcept { return _e.cols(); }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        return _op(_e[i], _v);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(std::size_t r, std::size_t c) const noexcept {
        return _op(_e(r, c), _v);
    }

private:
    const E& _e;
    T _v;
    Operator _op;
};

template <typename T, typename E, typename Operator>
class MatrixScalarLeft
    : public MatrixExpression<T, MatrixScalarLeft<T, E, Operator>> {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    MatrixScalarLeft(T v, const E& e, Operator op = Operator {}) noexcept
        : _v(v)
        , _e(e)
        , _op(op) { }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    rows() const noexcept { return _e.rows(); }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    cols() const noexcept { return _e.cols(); }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        return _op(_v, _e[i]);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(std::size_t r, std::size_t c) const noexcept {
        return _op(_v, _e(r, c));
    }

private:
    T _v;
    const E& _e;
    Operator _op;
};

template <typename T, typename EM, typename ET, typename EF>
class MatrixSelect
    : public MatrixExpression<T, MatrixSelect<T, EM, ET, EF>> {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    MatrixSelect(const EM& m, const ET& t, const EF& f) noexcept
        : _m(m)
        , _t(t)
        , _f(f) { }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    rows() const noexcept { return _t.rows(); }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    cols() const noexcept { return _t.cols(); }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {

        return _m[i] ? _t[i] : _f[i];
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(std::size_t r, std::size_t c) const noexcept {
        return _m(r, c) ? _t(r, c) : _f(r, c);
    }

private:
    const EM& _m;
    const ET& _t;
    const EF& _f;
};

template <typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
all(const MatrixExpression<bool, E>& mask) noexcept {
    const E& m          = mask();
    const std::size_t n = m.size();
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {

        if (!m[i]) return false;
    }
    return true;
}

template <typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
any(const MatrixExpression<bool, E>& mask) noexcept {
    const E& m          = mask();
    const std::size_t n = m.size();
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {

        if (m[i]) return true;
    }
    return false;
}
}