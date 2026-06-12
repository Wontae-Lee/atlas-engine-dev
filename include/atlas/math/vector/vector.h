#pragma once
#include <atlas/math/vector/vector_expression.h>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

namespace atlas {

    /**
     * @brief Fixed-size N-dimensional vector with expression-template support.
     *
     * @details
     * `Vector<T, N>` is a small, fixed-size vector type intended for math-heavy code on both
     * host and device. It stores `N` scalars in a contiguous, aligned array and provides
     * common vector operations such as element-wise arithmetic, dot product, norms,
     * normalization, reductions (sum/min/max), and axis queries (major/minor).
     *
     * The class participates in the expression-template system by inheriting from
     * `VectorExpression<T, Vector<T,N>>`, enabling lazy evaluation and efficient composition
     * of vector expressions.
     *
     * Storage is **contiguous** in memory (index order `[0..N-1]`) and aligned for
     * SIMD-friendly access:
     * - `data()` returns a pointer to the first element.
     * - `operator[]` and `at()` provide element access.
     *
     * ---
     *
     * @tparam T Scalar type. Must be trivially copyable.
     * @tparam N Vector dimension. Must be `N >= 1`.
     *
     * @note
     * - Equality uses exact element-wise comparison.
     * - `normalize()` and `normalized()` assume the length is non-zero; behavior for zero-length
     *   vectors depends on the implementation in `vector.hpp`.
     * - `major_axis()` / `minor_axis()` typically return the index of the component with the
     *   largest/smallest magnitude (implementation-defined tie-breaking).
     */
    template <typename T, std::size_t N>
    class Vector : public VectorExpression<T, Vector<T, N>> {
        static_assert(N >= 1, "Vector dimension must be >= 1");
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    public:
        /**
         * @brief Default constructor.
         *
         * @details
         * Initializes the vector to an implementation-defined state (commonly zero).
         * See `vector.hpp` for the exact behavior.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        Vector() noexcept;

        /// @brief Copy constructor.
        Vector(const Vector& other) noexcept = default;

        /**
         * @brief Constructs a vector with all components set to the same scalar.
         *
         * @param s Scalar value assigned to every component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Vector(T s) noexcept;

        /**
         * @brief Constructs a vector from exactly `N` component values.
         *
         * @details
         * The arguments are interpreted in index order:
         * \f[
         *   v = (a_0, a_1, \dots, a_{N-1})
         * \f]
         *
         * @tparam Args Variadic component types convertible to `T`.
         * @param args Exactly `N` values.
         */
        template <typename... Args,
                  typename = std::enable_if_t<(sizeof...(Args) == N)
                                              && (std::conjunction_v<std::is_convertible<Args, T>...>)>>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Vector(Args... args) noexcept;

        /**
         * @brief Constructs from an initializer list.
         *
         * @details
         * Values are copied in order into indices `[0..]`.
         * If fewer than `N` values are provided, remaining elements are filled in an
         * implementation-defined way (commonly zero). Extra values are typically ignored.
         *
         * @param list Initializer list of component values.
         */
        ATLAS_HOST ATLAS_FORCE_INLINE
        Vector(std::initializer_list<T> list) noexcept;

        /**
         * @brief Constructs from a vector expression.
         *
         * @details
         * Evaluates the expression and stores the result in this vector.
         *
         * @tparam Expression Expression type.
         * @param expr Expression to evaluate.
         */
        template <typename Expression>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        Vector(const VectorExpression<T, Expression>& expr) noexcept;

        ~Vector() noexcept = default;

        /**
         * @brief Returns the vector dimension `N`.
         *
         * @return Number of components.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        size() noexcept;

        /**
         * @brief Element access (const) by index.
         *
         * @param index Component index in `[0, N)`.
         * @return Const reference to the component.
         *
         * @note No bounds checking is implied.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        operator[](std::size_t index) const noexcept;

        /**
         * @brief Element access (mutable) by index.
         *
         * @param index Component index in `[0, N)`.
         * @return Reference to the component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        operator[](std::size_t index) noexcept;

        /**
         * @brief Copy assignment.
         *
         * @param rhs Right-hand side vector.
         * @return Reference to `*this`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator=(const Vector& rhs) noexcept;

        /**
         * @brief Sets all components to the same scalar.
         *
         * @param s Scalar value assigned to every component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set(T s) noexcept;

        /**
         * @brief Sets the vector from exactly `N` component values.
         *
         * @tparam Args Variadic component types convertible to `T`.
         * @param args Exactly `N` values.
         */
        template <typename... Args,
                  typename = std::enable_if_t<(sizeof...(Args) == N)
                                              && (std::conjunction_v<std::is_convertible<Args, T>...>)>>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_values(Args... args) noexcept;

        /**
         * @brief Sets all components to zero.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_zero() noexcept;

        /**
         * @brief Adds a scalar to all components in-place.
         *
         * @param v Scalar to add.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(T v) noexcept;

        /**
         * @brief Subtracts a scalar from all components in-place.
         *
         * @param v Scalar to subtract.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        sub(T v) noexcept;

        /**
         * @brief Multiplies all components by a scalar in-place.
         *
         * @param v Scalar multiplier.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        mul(T v) noexcept;

        /**
         * @brief Divides all components by a scalar in-place.
         *
         * @param v Scalar divisor.
         *
         * @note Division by zero behavior is implementation-defined.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        div(T v) noexcept;

        /**
         * @brief Adds another vector element-wise in-place.
         *
         * @param v Vector to add.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(const Vector& v) noexcept;

        /**
         * @brief Subtracts another vector element-wise in-place.
         *
         * @param v Vector to subtract.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        sub(const Vector& v) noexcept;

        /**
         * @brief Multiplies by another vector element-wise in-place (Hadamard product).
         *
         * @param v Vector to multiply component-wise.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        mul(const Vector& v) noexcept;

        /**
         * @brief Divides by another vector element-wise in-place.
         *
         * @param v Vector divisor (component-wise).
         *
         * @note Division by zero behavior is implementation-defined.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        div(const Vector& v) noexcept;

        /// @brief In-place scalar add.
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator+=(T v) noexcept;

        /// @brief In-place scalar subtract.
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator-=(T v) noexcept;

        /// @brief In-place scalar multiply.
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator*=(T v) noexcept;

        /// @brief In-place scalar divide.
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator/=(T v) noexcept;

        /// @brief In-place vector add (element-wise).
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator+=(const Vector& v) noexcept;

        /// @brief In-place vector subtract (element-wise).
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator-=(const Vector& v) noexcept;

        /// @brief In-place vector multiply (element-wise).
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator*=(const Vector& v) noexcept;

        /// @brief In-place vector divide (element-wise).
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator/=(const Vector& v) noexcept;

        /**
         * @brief Exact element-wise equality.
         *
         * @param other Vector to compare.
         * @return `true` if all components are exactly equal.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator==(const Vector& other) const noexcept;

        /**
         * @brief Computes the dot product with another vector.
         *
         * @details
         * \f[
         *   \mathbf{a}\cdot\mathbf{b}=\sum_{i=0}^{N-1} a_i b_i
         * \f]
         *
         * @param v Other vector.
         * @return Dot product.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        dot(const Vector& v) const noexcept;

        /**
         * @brief Returns the squared Euclidean length.
         *
         * @details
         * \f[
         *   \|\mathbf{v}\|^2=\mathbf{v}\cdot\mathbf{v}
         * \f]
         *
         * @return Squared length (non-negative).
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        length_squared() const noexcept;

        /**
         * @brief Returns the Euclidean length (L2 norm).
         *
         * @details
         * \f[
         *   \|\mathbf{v}\|=\sqrt{\mathbf{v}\cdot\mathbf{v}}
         * \f]
         *
         * @return Length (non-negative).
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        length() const noexcept;

        /**
         * @brief Normalizes this vector in-place.
         *
         * @details
         * Scales the vector so that its length becomes 1:
         * \f[
         *   \mathbf{v}\leftarrow \frac{\mathbf{v}}{\|\mathbf{v}\|}
         * \f]
         *
         * @note
         * Behavior for zero-length vectors is implementation-defined.
         * If you need safety, guard with `length_squared() > 0`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        normalize() noexcept;

        /**
         * @brief Returns a normalized copy of this vector.
         *
         * @return Normalized vector.
         *
         * @note
         * Behavior for zero-length vectors is implementation-defined.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector
        normalized() const noexcept;

        /**
         * @brief Casts the vector components to another scalar type.
         *
         * @tparam To Destination scalar type.
         * @return Vector with components converted to `To`.
         */
        template <typename To>
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<To, N>
        cast_to() const noexcept;

        /**
         * @brief Returns a pointer to contiguous storage (const).
         *
         * @return Pointer to the first element.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
        data() const noexcept;

        /**
         * @brief Returns a pointer to contiguous storage (mutable).
         *
         * @return Pointer to the first element.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
        data() noexcept;

        /**
         * @brief Bounds-access element getter (const).
         *
         * @details
         * Semantically equivalent to `operator[]`, but intended to be a safer accessor.
         * Whether it checks bounds depends on the implementation in `vector.hpp`.
         *
         * @param index Component index.
         * @return Const reference to the component.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        at(std::size_t index) const noexcept;

        /**
         * @brief Bounds-access element getter (mutable).
         *
         * @param index Component index.
         * @return Reference to the component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        at(std::size_t index) noexcept;

        /**
         * @brief Returns the sum of all components.
         *
         * @return \f$\sum_i v_i\f$.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        sum() const noexcept;

        /**
         * @brief Returns the arithmetic mean of all components.
         *
         * @details
         * \f[
         *   \mathrm{avg}(\mathbf{v})=\frac{1}{N}\sum_{i=0}^{N-1} v_i
         * \f]
         *
         * @return Average value.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        avg() const noexcept;

        /**
         * @brief Returns the minimum component value.
         *
         * @return \f$\min_i v_i\f$.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        min() const noexcept;

        /**
         * @brief Returns the maximum component value.
         *
         * @return \f$\max_i v_i\f$.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        max() const noexcept;

        /**
         * @brief Returns the index of the major axis.
         *
         * @details
         * Typically returns the component index with the largest magnitude
         * (implementation-defined whether magnitude or raw value is used).
         *
         * @return Index in `[0, N)`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
        major_axis() const noexcept;

        /**
         * @brief Returns the index of the minor axis.
         *
         * @details
         * Typically returns the component index with the smallest magnitude
         * (implementation-defined whether magnitude or raw value is used).
         *
         * @return Index in `[0, N)`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
        minor_axis() const noexcept;

    private:
        /// @brief Contiguous storage for `N` components (aligned for SIMD-friendly access).
        alignas(32) T _data[N];
    };
} // namespace atlas

#include <atlas/math/vector/vector.hpp>
