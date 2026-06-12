#pragma once

#include <atlas/math/detail/ops.h>
#include <cstddef>
#include <type_traits>

namespace atlas {

/**
 * @file vector_expression.h
 * @brief Core vector expression-template infrastructure (CRTP).
 *
 * @details
 * This header provides the minimal building blocks for Atlas' vector expression system:
 * - A CRTP base (`VectorExpression<T, E>`) that defines the common expression interface.
 * - Expression node types for unary and binary operations.
 * - A scalar-binary node to combine an expression with a scalar.
 * - A lane-wise select node driven by a boolean mask expression.
 *
 * The primary goal is **lazy evaluation**: most operations construct lightweight nodes that
 * reference existing expressions by const reference and only compute values when indexed
 * via `operator[]`.
 *
 * @note
 * - Expression nodes do not own underlying vector storage; they only store references.
 * - Callers must ensure referenced operands outlive the expression nodes (typical for
 *   temporary chaining within a full expression statement).
 * - All APIs are annotated with `ATLAS_ALL_DEVICE` so they can be used in host and device code.
 *
 * @see atlas/math/detail/ops.h for operator functors and value-type traits.
 */

// ------------------------------------------------------------
// Base tags / CRTP interface
// ------------------------------------------------------------

/**
 * @brief Marker base class used to tag expression families by scalar value type.
 *
 * @tparam T Scalar lane type of the expression (e.g., float, double, bool).
 *
 * @details
 * `VectorExpressionBase<T>` does not define an interface; it exists primarily to enable
 * type traits and concepts for expression-template detection.
 */
template <typename T>
class VectorExpressionBase { };

/**
 * @brief CRTP base class that defines the public interface of a vector expression.
 *
 * @tparam T Scalar lane type returned by `operator[]`.
 * @tparam E Derived expression node type (CRTP).
 *
 * @details
 * The derived type `E` must implement:
 * - `std::size_t size() const noexcept`
 * - `T operator[](std::size_t) const noexcept`
 *
 * This wrapper forwards calls to the derived object via `static_cast<const E&>(*this)`.
 *
 * @note
 * - `operator()()` exposes the derived expression reference; this is a common expression-template
 *   idiom enabling nested nodes to store operands as references without slicing.
 * - The interface is intentionally minimal to maximize inlining and backend fusion.
 */
template <typename T, typename E>
class VectorExpression : public VectorExpressionBase<T> {
public:
    /**
     * @brief Returns the number of lanes (elements) in this expression.
     *
     * @return Expression length.
     *
     * @note
     * For fixed-size vector types, compilers can constant-fold this value.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    size() const noexcept {
        // Forward to the derived node (no virtual dispatch).
        return static_cast<const E&>(*this).size();
    }

    /**
     * @brief Returns a const reference to the derived expression node.
     *
     * @return `static_cast<const E&>(*this)`.
     *
     * @details
     * This is used to "unwrap" CRTP types when building expression nodes:
     * `VectorUnaryOperator(..., expr())` stores a reference to the actual derived type.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const E&
    operator()() const noexcept {
        // Expose the derived node reference (CRTP unwrapping).
        return static_cast<const E&>(*this);
    }

    /**
     * @brief Returns the i-th element of the expression.
     *
     * @param i Lane index.
     * @return Lane value at index `i`.
     *
     * @note
     * No bounds checking is performed; callers are expected to respect `0 <= i < size()`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        // Forward to the derived node element accessor.
        return static_cast<const E&>(*this)[i];
    }
};

// ------------------------------------------------------------
// Traits / concepts
// ------------------------------------------------------------

/**
 * @brief Helper alias for the scalar lane type of an expression-like type `E`.
 *
 * @tparam E Expression node type.
 *
 * @details
 * Delegates to `detail::expr_value_t<E>` (defined in `atlas/math/detail/ops.h`),
 * which is expected to compute the "value type" associated with an expression node.
 */
template <typename E>
using expr_value_t = detail::expr_value_t<E>;

/**
 * @brief Concept checking whether a type models an Atlas vector expression node.
 *
 * @tparam E Candidate type.
 *
 * @details
 * A type satisfies `VectorExpressionType` if it derives from:
 * `VectorExpression<expr_value_t<E>, E>`.
 *
 * This is a common CRTP pattern: a derived node `E` inherits the interface via
 * `VectorExpression<T, E>`.
 */
template <typename E>
concept VectorExpressionType = std::is_base_of_v<VectorExpression<expr_value_t<E>, E>, E>;

// ------------------------------------------------------------
// Unary operator node
// ------------------------------------------------------------

/**
 * @brief Expression node that applies a unary operator per lane.
 *
 * @tparam T Result scalar type.
 * @tparam E Operand expression type.
 * @tparam Operator Functor type implementing `T operator()(operand_value) const`.
 *
 * @details
 * This node stores:
 * - a const reference to the operand expression (`_operand`)
 * - a copy of the operator functor (`_op`)
 *
 * The value is produced lazily in `operator[]` by calling `_op(_operand[i])`.
 *
 * @note
 * - The operand is stored by reference; ensure it outlives this node.
 * - The operator is stored by value to allow small stateless functors and also
 *   stateful functors (e.g., clamp bounds) to be captured.
 */
template <typename T, typename E, typename Operator>
class VectorUnaryOperator
    : public VectorExpression<T, VectorUnaryOperator<T, E, Operator>> {
public:
    /**
     * @brief Constructs a unary operator expression node.
     *
     * @param operand Operand expression to wrap.
     * @param op Operator functor (default constructed if omitted).
     *
     * @note
     * `operand` is stored by reference; `op` is stored by value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit VectorUnaryOperator(const E& operand, Operator op = Operator {}) noexcept
        : _operand(operand) // Store operand by reference (no copy of underlying data).
        , _op(op)           // Store functor by value (may contain parameters).
    { }

    /**
     * @brief Returns the number of lanes in the expression.
     *
     * @return Operand size.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    size() const noexcept {
        // Unary node inherits size from its operand.
        return _operand.size();
    }

    /**
     * @brief Evaluates the unary operation at lane `i`.
     *
     * @param i Lane index.
     * @return `_op(_operand[i])`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        // Lazy evaluation: compute only the requested lane.
        return _op(_operand[i]);
    }

private:
    const E& _operand; // Operand expression (referenced, not owned).
    Operator _op;      // Operator functor (owned by value).
};

// ------------------------------------------------------------
// Binary operator node
// ------------------------------------------------------------

/**
 * @brief Expression node that applies a binary operator per lane.
 *
 * @tparam T Result scalar type.
 * @tparam EL Left operand expression type.
 * @tparam ER Right operand expression type.
 * @tparam Operator Functor type implementing `T operator()(left_value, right_value) const`.
 *
 * @details
 * Stores const references to both operands and a copy of the operator functor.
 * Evaluation is lane-wise and lazy:
 * `operator[](i)` returns `_op(_l[i], _r[i])`.
 *
 * @note
 * - No size reconciliation is performed; it is assumed both operands have the same size.
 * - The size of the expression is taken from the left operand (`_l.size()`).
 */
template <typename T, typename EL, typename ER, typename Operator>
class VectorBinaryOperator
    : public VectorExpression<T, VectorBinaryOperator<T, EL, ER, Operator>> {
public:
    /**
     * @brief Constructs a binary operator expression node.
     *
     * @param l Left operand expression.
     * @param r Right operand expression.
     * @param op Operator functor (default constructed if omitted).
     *
     * @note
     * `l` and `r` are stored by reference; `op` is stored by value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    VectorBinaryOperator(const EL& l, const ER& r, Operator op = Operator {}) noexcept
        : _l(l)   // Store left operand by reference.
        , _r(r)   // Store right operand by reference.
        , _op(op) // Store functor by value.
    { }

    /**
     * @brief Returns the number of lanes in the expression.
     *
     * @return Left operand size.
     *
     * @note
     * Assumes `l.size() == r.size()`. Mismatched sizes are undefined behavior.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    size() const noexcept {
        // Convention: take size from the left operand.
        return _l.size();
    }

    /**
     * @brief Evaluates the binary operation at lane `i`.
     *
     * @param i Lane index.
     * @return `_op(_l[i], _r[i])`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        // Lazy evaluation: compute only one lane, fusing left/right reads and op application.
        return _op(_l[i], _r[i]);
    }

private:
    const EL& _l; // Left operand (referenced, not owned).
    const ER& _r; // Right operand (referenced, not owned).
    Operator _op; // Operator functor (owned by value).
};

// ------------------------------------------------------------
// Expression-scalar binary operator node
// ------------------------------------------------------------

/**
 * @brief Expression node combining an expression with a scalar via a binary operator.
 *
 * @tparam T Result scalar type (and scalar operand type).
 * @tparam E Expression operand type.
 * @tparam Operator Functor implementing `T operator()(expr_value, scalar) const`.
 *
 * @details
 * This is useful for operations like `expr + scalar`, `expr * scalar`, etc.
 * The scalar is stored by value (`_v`) so it remains valid even if the caller
 * passes a temporary.
 *
 * @note
 * - The expression operand is stored by reference; ensure it outlives this node.
 * - The scalar is captured by value to avoid dangling references.
 */
template <typename T, typename E, typename Operator>
class VectorScalarBinaryOperator
    : public VectorExpression<T, VectorScalarBinaryOperator<T, E, Operator>> {
public:
    /**
     * @brief Constructs a scalar-binary operator expression node.
     *
     * @param e Expression operand.
     * @param v Scalar value (copied).
     * @param op Operator functor (default constructed if omitted).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    VectorScalarBinaryOperator(const E& e, const T& v, Operator op = Operator {}) noexcept
        : _e(e)   // Store expression by reference.
        , _v(v)   // Store scalar by value (safe for temporaries).
        , _op(op) // Store functor by value.
    { }

    /**
     * @brief Returns the number of lanes in the expression.
     *
     * @return Expression operand size.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    size() const noexcept {
        // Size matches the expression operand.
        return _e.size();
    }

    /**
     * @brief Evaluates the operation at lane `i`.
     *
     * @param i Lane index.
     * @return `_op(_e[i], _v)`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        // Apply op using per-lane expression value and the captured scalar.
        return _op(_e[i], _v);
    }

private:
    const E& _e;  // Expression operand (referenced, not owned).
    T _v;         // Scalar operand (owned by value).
    Operator _op; // Operator functor (owned by value).
};

// ------------------------------------------------------------
// Lane-wise select node
// ------------------------------------------------------------

/**
 * @brief Expression node selecting between two expressions per lane using a mask.
 *
 * @tparam T Result scalar type.
 * @tparam EM Mask expression type (expects `operator[](i)` convertible to bool).
 * @tparam ET True-branch expression type.
 * @tparam EF False-branch expression type.
 *
 * @details
 * For each lane `i`, returns:
 * - `_t[i]` if `_m[i]` is true
 * - `_f[i]` otherwise
 *
 * This is a building block for branchless conditionals in expression graphs.
 *
 * @note
 * - The mask and both branches are stored by reference; ensure they outlive this node.
 * - The size is taken from the true branch (`_t.size()`), assuming all operands match.
 */
template <typename T, typename EM, typename ET, typename EF>
class VectorSelect
    : public VectorExpression<T, VectorSelect<T, EM, ET, EF>> {
public:
    /**
     * @brief Constructs a select expression node.
     *
     * @param m Mask expression.
     * @param t Expression selected when mask lane is true.
     * @param f Expression selected when mask lane is false.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    VectorSelect(const EM& m, const ET& t, const EF& f) noexcept
        : _m(m) // Store mask by reference.
        , _t(t) // Store true branch by reference.
        , _f(f) // Store false branch by reference.
    { }

    /**
     * @brief Returns the number of lanes in the expression.
     *
     * @return True-branch size.
     *
     * @note
     * Assumes `t.size() == f.size()` and compatible mask size.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    size() const noexcept {
        // Convention: use true-branch size as canonical.
        return _t.size();
    }

    /**
     * @brief Evaluates the select at lane `i`.
     *
     * @param i Lane index.
     * @return `_m[i] ? _t[i] : _f[i]`.
     *
     * @note
     * This is a per-lane conditional; backends may lower it to predication/branchless ops.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    operator[](std::size_t i) const noexcept {
        // Evaluate mask lane and select the corresponding branch lane.
        return _m[i] ? _t[i] : _f[i];
    }

private:
    const EM& _m; // Mask expression (referenced, not owned).
    const ET& _t; // True branch expression (referenced, not owned).
    const EF& _f; // False branch expression (referenced, not owned).
};

} // namespace atlas
