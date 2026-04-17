#pragma once
#include <atlas/math/detail/ops.h>
#include <cstddef>
#include <type_traits>
namespace atlas::math {
/**
 * @file matrix_expression.h
 * @brief Core matrix expression-template infrastructure (CRTP) and boolean reductions.
 *
 * @details
 * This header provides the foundational building blocks for Atlas' matrix expression system:
 * - A CRTP interface (`MatrixExpression<T, E>`) that forwards shape/element access to the derived node.
 * - Expression node types for unary and binary element-wise operations.
 * - Scalar-element-wise nodes for `expr op scalar` and `scalar op expr`.
 * - A mask-based select node (`MatrixSelect`) for lane-wise branching.
 * - Boolean reductions (`all`, `any`) that evaluate mask expressions to scalar `bool`.
 *
 * The expression system is designed for **lazy evaluation**:
 * most operations build lightweight nodes holding references to operands, and compute values
 * only when indexed (`operator[]`) or addressed (`operator()(r,c)`).
 *
 * @note
 * - Expression nodes store operands by const reference; the caller must ensure referenced
 *   objects outlive the expression node.
 * - Scalar operands are stored by value to safely capture temporaries.
 * - Functions are annotated with `ATLAS_ALL_DEVICE` so they can be used in both host and device code.
 *
 * @see atlas/math/detail/ops.h for operator functors and expression value-type traits.
 */
// ------------------------------------------------------------
// Base tags / CRTP interface
// ------------------------------------------------------------
/**
 * @brief Marker base class used to tag matrix expressions by scalar value type.
 *
 * @tparam T Scalar element type (e.g., float, double, bool).
 *
 * @details
 * This base type does not define behavior; it exists to support type traits and concepts.
 */
template <typename T>
class MatrixExpressionBase { };
/**
 * @brief CRTP base class defining the public interface of a matrix expression.
 *
 * @tparam T Scalar element type returned by element accessors.
 * @tparam E Derived expression node type (CRTP).
 *
 * @details
 * The derived type `E` must provide:
 * - `std::size_t rows() const noexcept`
 * - `std::size_t cols() const noexcept`
 * - `T operator[](std::size_t) const noexcept`   (linear access)
 * - `T operator()(std::size_t, std::size_t) const noexcept` (2D access)
 *
 * This wrapper forwards to the derived type via `static_cast<const E&>(*this)`,
 * enabling static polymorphism (no virtual dispatch).
 *
 * @note
 * `operator()()` returns a reference to the derived node and is used to "unwrap" CRTP types
 * when building higher-level expression nodes without slicing.
 */
template <typename T, typename E>
class MatrixExpression : public MatrixExpressionBase<T> {
public:
    /**
     * @brief Returns the number of rows of the expression.
     *
     * @return Row count.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    rows() const noexcept {
        // Forward to the derived node (CRTP).
        return static_cast<const E&>(*this).rows();
    }
    /**
     * @brief Returns the number of columns of the expression.
     *
     * @return Column count.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    cols() const noexcept {
        // Forward to the derived node (CRTP).
        return static_cast<const E&>(*this).cols();
    }
    /**
     * @brief Returns the total number of elements (`rows() * cols()`).
     *
     * @return Total element count.
     *
     * @note
     * This is a convenience helper; it does not imply contiguity or storage layout.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    size() const noexcept {
        // Compute total elements from shape.
        return rows() * cols();
    }
    /**
     * @brief Linear element access.
     *
     * @param i Linear element index.
     * @return Element value at linear index `i`.
     *
     * @note
     * No bounds checking is performed; caller must ensure `0 <= i < size()`.
     * The interpretation of linear indexing is defined by the derived node.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        // Forward linear access to the derived node.
        return static_cast<const E&>(*this)[i];
    }
    /**
     * @brief 2D element access.
     *
     * @param r Row index.
     * @param c Column index.
     * @return Element value at (r, c).
     *
     * @note
     * No bounds checking is performed; caller must ensure `r < rows()` and `c < cols()`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(std::size_t r, std::size_t c) const noexcept {
        // Forward 2D access to the derived node.
        return static_cast<const E&>(*this)(r, c);
    }
    /**
     * @brief Returns a const reference to the derived expression node.
     *
     * @return `static_cast<const E&>(*this)`.
     *
     * @details
     * Used for CRTP unwrapping when building expression nodes that store operand references.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const E&
    operator()() const noexcept {
        // Expose derived node reference.
        return static_cast<const E&>(*this);
    }
};
// ------------------------------------------------------------
// Traits / concepts
// ------------------------------------------------------------
/**
 * @brief Helper alias for the scalar element type of an expression-like type `E`.
 *
 * @tparam E Candidate expression type.
 *
 * @details
 * Delegates to `detail::expr_value_t<E>` from `atlas/math/detail/ops.h`.
 */
template <typename E>
using expr_value_t = detail::expr_value_t<E>;
/**
 * @brief Concept checking whether a type is a matrix expression node (CRTP derived).
 *
 * @tparam E Candidate type.
 *
 * @details
 * Satisfied when `E` derives from `MatrixExpression<expr_value_t<E>, E>`.
 */
template <typename E>
concept MatrixExpressionType = std::is_base_of_v<MatrixExpression<expr_value_t<E>, E>, E>;
// ------------------------------------------------------------
// Unary operator node
// ------------------------------------------------------------
/**
 * @brief Expression node applying a unary operator to each matrix element.
 *
 * @tparam T Result element type.
 * @tparam E Operand expression type.
 * @tparam Operator Functor implementing `T operator()(operand_value) const`.
 *
 * @details
 * Stores:
 * - a const reference to the operand expression (`_e`)
 * - a copy of the operator functor (`_op`)
 *
 * Values are computed lazily in `operator[]` or `operator()(r,c)`.
 *
 * @note
 * Operand is stored by reference; ensure it outlives this node.
 * Operator is stored by value (allows capturing parameters such as clamp bounds).
 */
template <typename T, typename E, typename Operator>
class MatrixUnaryOperator
    : public MatrixExpression<T, MatrixUnaryOperator<T, E, Operator>> {
public:
    /**
     * @brief Constructs a unary operator node.
     *
     * @param operand Operand expression (stored by reference).
     * @param op Operator functor (stored by value; default constructed if omitted).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit MatrixUnaryOperator(const E& operand, Operator op = Operator {}) noexcept
        : _e(operand) // Reference to operand expression (no ownership).
        , _op(op)     // Operator functor (owned).
    { }

    /**
     * @brief Returns the row count of the expression.
     *
     * @return Operand rows.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    rows() const noexcept { return _e.rows(); }

    /**
     * @brief Returns the column count of the expression.
     *
     * @return Operand cols.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    cols() const noexcept { return _e.cols(); }

    /**
     * @brief Evaluates the unary op at linear index `i`.
     *
     * @param i Linear element index.
     * @return `_op(_e[i])`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        // Lazy evaluation per element.
        return _op(_e[i]);
    }

    /**
     * @brief Evaluates the unary op at element (r, c).
     *
     * @param r Row index.
     * @param c Column index.
     * @return `_op(_e(r,c))`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(std::size_t r, std::size_t c) const noexcept {
        // Lazy evaluation per element using 2D access.
        return _op(_e(r, c));
    }

private:
    const E& _e;  // Operand expression (referenced, not owned).
    Operator _op; // Operator functor (owned by value).
};
// ------------------------------------------------------------
// Binary operator node
// ------------------------------------------------------------
/**
 * @brief Expression node applying a binary operator to each element pair from two expressions.
 *
 * @tparam T Result element type.
 * @tparam EL Left operand expression type.
 * @tparam ER Right operand expression type.
 * @tparam Operator Functor implementing `T operator()(left_value, right_value) const`.
 *
 * @details
 * The node stores references to both operands and applies the functor element-wise.
 * Shape is taken from the left operand.
 *
 * @warning
 * This assumes both operands have identical shape; no runtime checks are performed.
 */
template <typename T, typename EL, typename ER, typename Operator>
class MatrixBinaryOperator
    : public MatrixExpression<T, MatrixBinaryOperator<T, EL, ER, Operator>> {
public:
    /**
     * @brief Constructs a binary operator node.
     *
     * @param l Left operand (stored by reference).
     * @param r Right operand (stored by reference).
     * @param op Operator functor (stored by value; default constructed if omitted).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    MatrixBinaryOperator(const EL& l, const ER& r, Operator op = Operator {}) noexcept
        : _l(l)   // Reference to left operand.
        , _r(r)   // Reference to right operand.
        , _op(op) // Operator functor.
    { }

    /** @brief Returns row count (from left operand). */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    rows() const noexcept { return _l.rows(); }

    /** @brief Returns column count (from left operand). */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    cols() const noexcept { return _l.cols(); }

    /**
     * @brief Evaluates the binary op at linear index `i`.
     *
     * @param i Linear element index.
     * @return `_op(_l[i], _r[i])`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        // Lazy evaluation; reads both operands at lane i.
        return _op(_l[i], _r[i]);
    }

    /**
     * @brief Evaluates the binary op at element (r, c).
     *
     * @param r Row index.
     * @param c Column index.
     * @return `_op(_l(r,c), _r(r,c))`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(std::size_t r, std::size_t c) const noexcept {
        // Lazy evaluation using 2D element access.
        return _op(_l(r, c), _r(r, c));
    }

private:
    const EL& _l; // Left operand expression (referenced).
    const ER& _r; // Right operand expression (referenced).
    Operator _op; // Operator functor (owned).
};
// ------------------------------------------------------------
// Scalar (right) node: expr op scalar
// ------------------------------------------------------------
/**
 * @brief Expression node applying a binary operator between an expression element and a scalar (right operand).
 *
 * @tparam T Result element type (and scalar type).
 * @tparam E Expression operand type.
 * @tparam Operator Functor implementing `T operator()(expr_value, scalar) const`.
 *
 * @details
 * Represents operations of the form: `expr (op) scalar`, evaluated element-wise.
 * The scalar is stored by value so it is safe to pass temporaries.
 *
 * @note
 * The expression operand is stored by reference; ensure it outlives this node.
 */
template <typename T, typename E, typename Operator>
class MatrixScalarRight
    : public MatrixExpression<T, MatrixScalarRight<T, E, Operator>> {
public:
    /**
     * @brief Constructs a scalar-right operator node.
     *
     * @param e Expression operand (stored by reference).
     * @param v Scalar value (stored by value).
     * @param op Operator functor (stored by value; default constructed if omitted).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    MatrixScalarRight(const E& e, T v, Operator op = Operator {}) noexcept
        : _e(e)   // Reference to expression operand.
        , _v(v)   // Copy of scalar operand (safe).
        , _op(op) // Operator functor.
    { }

    /** @brief Returns row count from expression operand. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    rows() const noexcept { return _e.rows(); }

    /** @brief Returns column count from expression operand. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    cols() const noexcept { return _e.cols(); }

    /**
     * @brief Evaluates the op at linear index `i`.
     *
     * @param i Linear element index.
     * @return `_op(_e[i], _v)`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        return _op(_e[i], _v);
    }

    /**
     * @brief Evaluates the op at element (r, c).
     *
     * @param r Row index.
     * @param c Column index.
     * @return `_op(_e(r,c), _v)`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(std::size_t r, std::size_t c) const noexcept {
        return _op(_e(r, c), _v);
    }

private:
    const E& _e;  // Expression operand (referenced).
    T _v;         // Scalar operand (owned).
    Operator _op; // Operator functor (owned).
};
// ------------------------------------------------------------
// Scalar (left) node: scalar op expr
// ------------------------------------------------------------
/**
 * @brief Expression node applying a binary operator between a scalar (left operand) and an expression element.
 *
 * @tparam T Result element type (and scalar type).
 * @tparam E Expression operand type.
 * @tparam Operator Functor implementing `T operator()(scalar, expr_value) const`.
 *
 * @details
 * Represents operations of the form: `scalar (op) expr`, evaluated element-wise.
 * The scalar is stored by value; the expression is stored by reference.
 */
template <typename T, typename E, typename Operator>
class MatrixScalarLeft
    : public MatrixExpression<T, MatrixScalarLeft<T, E, Operator>> {
public:
    /**
     * @brief Constructs a scalar-left operator node.
     *
     * @param v Scalar value (stored by value).
     * @param e Expression operand (stored by reference).
     * @param op Operator functor (stored by value; default constructed if omitted).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    MatrixScalarLeft(T v, const E& e, Operator op = Operator {}) noexcept
        : _v(v)   // Copy of scalar.
        , _e(e)   // Reference to expression.
        , _op(op) // Operator functor.
    { }

    /** @brief Returns row count from expression operand. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    rows() const noexcept { return _e.rows(); }

    /** @brief Returns column count from expression operand. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    cols() const noexcept { return _e.cols(); }

    /**
     * @brief Evaluates the op at linear index `i`.
     *
     * @param i Linear element index.
     * @return `_op(_v, _e[i])`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        return _op(_v, _e[i]);
    }

    /**
     * @brief Evaluates the op at element (r, c).
     *
     * @param r Row index.
     * @param c Column index.
     * @return `_op(_v, _e(r,c))`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(std::size_t r, std::size_t c) const noexcept {
        return _op(_v, _e(r, c));
    }

private:
    T _v;         // Scalar operand (owned).
    const E& _e;  // Expression operand (referenced).
    Operator _op; // Operator functor (owned).
};
// ------------------------------------------------------------
// Select node
// ------------------------------------------------------------
/**
 * @brief Expression node selecting element-wise between two expressions using a mask.
 *
 * @tparam T Result element type.
 * @tparam EM Mask expression type (expects `operator[]` and `operator()(r,c)` convertible to bool).
 * @tparam ET True-branch expression type.
 * @tparam EF False-branch expression type.
 *
 * @details
 * For each element:
 * - returns `_t(...)` if `_m(...)` is true
 * - otherwise returns `_f(...)`
 *
 * Shape is taken from the true-branch expression.
 *
 * @warning
 * Assumes mask and both branches have compatible shapes; no runtime validation is performed.
 */
template <typename T, typename EM, typename ET, typename EF>
class MatrixSelect
    : public MatrixExpression<T, MatrixSelect<T, EM, ET, EF>> {
public:
    /**
     * @brief Constructs a select node.
     *
     * @param m Mask expression (stored by reference).
     * @param t True-branch expression (stored by reference).
     * @param f False-branch expression (stored by reference).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    MatrixSelect(const EM& m, const ET& t, const EF& f) noexcept
        : _m(m) // Reference to mask.
        , _t(t) // Reference to true-branch.
        , _f(f) // Reference to false-branch.
    { }

    /** @brief Returns row count from the true-branch expression. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    rows() const noexcept { return _t.rows(); }

    /** @brief Returns column count from the true-branch expression. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    cols() const noexcept { return _t.cols(); }

    /**
     * @brief Evaluates the select at linear index `i`.
     *
     * @param i Linear element index.
     * @return `_m[i] ? _t[i] : _f[i]`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        // Lane-wise branching; backends may lower to predication.
        return _m[i] ? _t[i] : _f[i];
    }

    /**
     * @brief Evaluates the select at element (r, c).
     *
     * @param r Row index.
     * @param c Column index.
     * @return `_m(r,c) ? _t(r,c) : _f(r,c)`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator()(std::size_t r, std::size_t c) const noexcept {
        return _m(r, c) ? _t(r, c) : _f(r, c);
    }

private:
    const EM& _m; // Mask expression (referenced).
    const ET& _t; // True branch (referenced).
    const EF& _f; // False branch (referenced).
};
// ------------------------------------------------------------
// Boolean reductions (mask -> scalar)
// ------------------------------------------------------------
/**
 * @brief Returns true if all elements of a boolean mask matrix are true.
 *
 * @tparam E Mask expression type.
 * @param mask Boolean matrix expression.
 * @return `true` if every element is true; otherwise `false`.
 *
 * @details
 * Performs a short-circuiting reduction over the linear indexing domain:
 * returns immediately on the first false element.
 *
 * @note
 * Uses `ATLAS_UNROLL` to encourage unrolling for small fixed-size matrices.
 */
template <typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
all(const MatrixExpression<bool, E>& mask) noexcept {
    const E& m          = mask();   // Bind to derived mask node (no copy).
    const std::size_t n = m.size(); // Total number of elements.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        // Early-out on first false.
        if (!m[i]) return false;
    }
    return true; // All elements are true.
}
/**
 * @brief Returns true if any element of a boolean mask matrix is true.
 *
 * @tparam E Mask expression type.
 * @param mask Boolean matrix expression.
 * @return `true` if at least one element is true; otherwise `false`.
 *
 * @details
 * Performs a short-circuiting reduction over all elements:
 * returns immediately on the first true element.
 *
 * @note
 * Uses `ATLAS_UNROLL` to encourage unrolling for small fixed-size matrices.
 */
template <typename E>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
any(const MatrixExpression<bool, E>& mask) noexcept {
    const E& m          = mask();   // Bind to derived mask node.
    const std::size_t n = m.size(); // Total number of elements.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < n; ++i) {
        // Early-out on first true.
        if (m[i]) return true;
    }
    return false; // No elements were true.
}
} // namespace atlas::math
