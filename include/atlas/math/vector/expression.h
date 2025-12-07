#ifndef INCLUDE_ATLAS_MATH_VECTOR_EXPRESSION_H
#define INCLUDE_ATLAS_MATH_VECTOR_EXPRESSION_H
#include <atlas/math/detail/config.h>
#include <atlas/math/detail/ops.h>
#include <cstddef>
#include <type_traits>
namespace atlas::math {
template <typename T>
class VectorExpressionBase { };
template <typename T, typename E>
class VectorExpression : public VectorExpressionBase<T> {
public:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        std::size_t
        size() const noexcept { return static_cast<const E&>(*this).size(); }
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const E&
    operator()() const noexcept { return static_cast<const E&>(*this); }
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        T
        operator[](std::size_t i) const noexcept { return static_cast<const E&>(*this)[i]; }
};
template <typename E>
using expr_value_t = detail::expr_value_t<E>;
template <typename E>
concept VectorExpressionType = std::is_base_of_v<VectorExpression<expr_value_t<E>, E>, E>;
template <typename T, typename E, typename Operator>
class VectorUnaryOperator : public VectorExpression<T, VectorUnaryOperator<T, E, Operator>> {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit VectorUnaryOperator(const E& operand, Operator op = Operator {}) noexcept
        : _operand(operand)
        , _op(op) { }
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        std::size_t
        size() const noexcept { return _operand.size(); }
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        T
        operator[](std::size_t i) const noexcept { return _op(_operand[i]); }

private:
    const E& _operand;
    Operator _op;
};
template <typename T, typename EL, typename ER, typename Operator>
class VectorBinaryOperator : public VectorExpression<T, VectorBinaryOperator<T, EL, ER, Operator>> {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    VectorBinaryOperator(const EL& l, const ER& r, Operator op = Operator {}) noexcept
        : _l(l)
        , _r(r)
        , _op(op) { }
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        std::size_t
        size() const noexcept { return _l.size(); }
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        T
        operator[](std::size_t i) const noexcept { return _op(_l[i], _r[i]); }

private:
    const EL& _l;
    const ER& _r;
    Operator _op;
};
template <typename T, typename E, typename Operator>
class VectorScalarBinaryOperator : public VectorExpression<T, VectorScalarBinaryOperator<T, E, Operator>> {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    VectorScalarBinaryOperator(const E& e, const T& v, Operator op = Operator {}) noexcept
        : _e(e)
        , _v(v)
        , _op(op) { }
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        std::size_t
        size() const noexcept { return _e.size(); }
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        T
        operator[](std::size_t i) const noexcept { return _op(_e[i], _v); }

private:
    const E& _e;
    T _v;
    Operator _op;
};
template <typename T, typename EM, typename ET, typename EF>
class VectorSelect : public VectorExpression<T, VectorSelect<T, EM, ET, EF>> {
public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    VectorSelect(const EM& m, const ET& t, const EF& f) noexcept
        : _m(m)
        , _t(t)
        , _f(f) { }
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        std::size_t
        size() const noexcept { return _t.size(); }
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        T
        operator[](std::size_t i) const noexcept { return _m[i] ? _t[i] : _f[i]; }

private:
    const EM& _m;
    const ET& _t;
    const EF& _f;
};
}
#endif