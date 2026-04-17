#pragma once

#include <atlas/math/vector/vector.h>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

namespace atlas {
namespace math {

    /**
     * @brief Specialized 4D vector with named components and common vector operations.
     *
     * @details
     * This is a partial specialization of `Vector<T, N>` for `N = 4`, providing:
     * - Named fields `x`, `y`, `z`, `w` (and index-based access via `operator[]` / `at()`).
     * - Standard operations such as dot product, length/normalization, reflection/projection,
     *   and component-wise utility functions.
     *
     * The class participates in Atlas' vector expression system by inheriting from
     * `VectorExpression<T, Vector<T,4>>`, enabling efficient evaluation of expression trees.
     *
     * Storage semantics:
     * - `x` corresponds to index 0
     * - `y` corresponds to index 1
     * - `z` corresponds to index 2
     * - `w` corresponds to index 3
     * - `data()` returns a pointer to `x` (contiguous four-element storage)
     *
     * ---
     *
     * @tparam T Scalar type (must be trivially copyable).
     *
     * @note
     * - Equality is exact element-wise comparison (no epsilon).
     * - `normalize()` assumes non-zero length; zero-length handling is implementation-defined.
     * - `reflected()` / `projected()` typically assume the normal direction is unit length for
     *   geometric correctness.
     */
    template <typename T>
    class Vector<T, 4> : public VectorExpression<T, Vector<T, 4>> {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    public:
        /// @brief X component (index 0).
        T x;
        /// @brief Y component (index 1).
        T y;
        /// @brief Z component (index 2).
        T z;
        /// @brief W component (index 3).
        T w;

        /**
         * @brief Default constructor.
         *
         * @details
         * Initializes the vector to an implementation-defined state (commonly `{0,0,0,0}`).
         * See `vector4.hpp` for exact behavior.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Vector() noexcept;

        /// @brief Copy constructor.
        constexpr Vector(const Vector& v) noexcept = default;

        /**
         * @brief Constructs a vector with all components set to the same scalar.
         *
         * @param s Scalar assigned to `x`, `y`, `z`, and `w`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Vector(T s) noexcept;

        /**
         * @brief Constructs a vector from explicit components.
         *
         * @param x_ X component.
         * @param y_ Y component.
         * @param z_ Z component.
         * @param w_ W component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Vector(T x_, T y_, T z_, T w_) noexcept;

        /**
         * @brief Constructs from an initializer list.
         *
         * @details
         * Expected order is `{x, y, z, w}`. If fewer than 4 values are provided, remaining
         * components are filled in an implementation-defined way (commonly zero).
         * Extra values are typically ignored.
         *
         * @param list Initializer list of components.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Vector(std::initializer_list<T> list) noexcept;

        /**
         * @brief Constructs from a vector expression.
         *
         * @tparam Expression Expression type.
         * @param expr Expression to evaluate into this vector.
         */
        template <typename Expression>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Vector(const VectorExpression<T, Expression>& expr) noexcept;

        ~Vector() noexcept = default;

        /**
         * @brief Returns the dimension of the vector (always 4).
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        size() noexcept;

        /**
         * @brief Returns a pointer to contiguous storage (const).
         *
         * @return Pointer to `x`, followed by `y`, `z`, then `w`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
        data() const noexcept;

        /**
         * @brief Returns a pointer to contiguous storage (mutable).
         *
         * @return Pointer to `x`, followed by `y`, `z`, then `w`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
        data() noexcept;

        /**
         * @brief Element access (const) by index.
         *
         * @param i Index in `{0,1,2,3}` (`0 → x`, `1 → y`, `2 → z`, `3 → w`).
         * @return Const reference to the component.
         *
         * @note No bounds checking is implied.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        operator[](std::size_t i) const noexcept;

        /**
         * @brief Element access (mutable) by index.
         *
         * @param i Index in `{0,1,2,3}` (`0 → x`, `1 → y`, `2 → z`, `3 → w`).
         * @return Reference to the component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        operator[](std::size_t i) noexcept;

        /**
         * @brief Element access (const), intended as a safer accessor.
         *
         * @param i Index in `{0,1,2,3}`.
         * @return Const reference to the component.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        at(std::size_t i) const noexcept;

        /**
         * @brief Element access (mutable), intended as a safer accessor.
         *
         * @param i Index in `{0,1,2,3}`.
         * @return Reference to the component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        at(std::size_t i) noexcept;

        /**
         * @brief Sets all components to the same scalar.
         *
         * @param s Scalar to assign to `x`, `y`, `z`, and `w`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set(T s) noexcept;

        /**
         * @brief Sets all components from four values.
         *
         * @tparam Args Must be exactly 4 arguments convertible to `T`.
         * @param args Values assigned in order: `x`, `y`, `z`, `w`.
         */
        template <typename... Args,
                  typename = std::enable_if_t<(sizeof...(Args) == 4)
                                              && (std::conjunction_v<std::is_convertible<Args, T>...>)>>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_values(Args... args) noexcept;

        /** @brief Sets all components to zero. */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_zero() noexcept;

        /** @brief Adds a scalar to all components in-place. */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(T v) noexcept;

        /** @brief Subtracts a scalar from all components in-place. */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        sub(T v) noexcept;

        /** @brief Multiplies all components by a scalar in-place. */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        mul(T v) noexcept;

        /**
         * @brief Divides all components by a scalar in-place.
         *
         * @note Division by zero behavior is implementation-defined.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        div(T v) noexcept;

        /** @brief Adds another vector element-wise in-place. */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(const Vector& v) noexcept;

        /** @brief Subtracts another vector element-wise in-place. */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        sub(const Vector& v) noexcept;

        /** @brief Multiplies by another vector element-wise in-place (Hadamard product). */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        mul(const Vector& v) noexcept;

        /**
         * @brief Divides by another vector element-wise in-place.
         *
         * @note Division by zero behavior is implementation-defined.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        div(const Vector& v) noexcept;

        /**
         * @brief Returns the minimum of the four component values.
         *
         * @return `min(x, y, z, w)`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        min() const noexcept;

        /**
         * @brief Returns the maximum of the four component values.
         *
         * @return `max(x, y, z, w)`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        max() const noexcept;

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

        /** @brief Exact element-wise equality. */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator==(const Vector& other) const noexcept;

        /** @brief Exact element-wise inequality. */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator!=(const Vector& other) const noexcept;

        /**
         * @brief Computes the dot product with another 4D vector.
         *
         * @details
         * \f[
         * (x_1,y_1,z_1,w_1)\cdot(x_2,y_2,z_2,w_2)=x_1x_2+y_1y_2+z_1z_2+w_1w_2
         * \f]
         *
         * @param v Other vector.
         * @return Dot product.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        dot(const Vector& v) const noexcept;

        /** @brief Returns the squared Euclidean length. */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        length_squared() const noexcept;

        /** @brief Returns the Euclidean length. */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        length() const noexcept;

        /**
         * @brief Returns the index of the major axis.
         *
         * @details
         * Typically returns the component index with the largest absolute value
         * (implementation-defined tie-breaking).
         *
         * @return Index in `{0,1,2,3}`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
        major_axis() const noexcept;

        /**
         * @brief Returns the index of the minor axis.
         *
         * @details
         * Typically returns the component index with the smallest absolute value
         * (implementation-defined tie-breaking).
         *
         * @return Index in `{0,1,2,3}`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
        minor_axis() const noexcept;

        /**
         * @brief Normalizes this vector in-place.
         *
         * @details
         * \f[
         * \mathbf{v}\leftarrow \frac{\mathbf{v}}{\|\mathbf{v}\|}
         * \f]
         *
         * @note Behavior for zero-length vectors is implementation-defined.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        normalize() noexcept;

        /**
         * @brief Returns a normalized copy of this vector.
         *
         * @note Behavior for zero-length vectors is implementation-defined.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector
        normalized() const noexcept;

        /**
         * @brief Returns the reflection of this vector about a (unit) normal.
         *
         * @details
         * Reflection formula (applied in 4D component space):
         * \f[
         * \mathbf{r} = \mathbf{v} - 2(\mathbf{v}\cdot\mathbf{n})\mathbf{n}
         * \f]
         *
         * @param n Normal direction (typically unit length).
         * @return Reflected vector.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector
        reflected(const Vector& n) const noexcept;

        /**
         * @brief Projects this vector onto direction `n`.
         *
         * @details
         * Projection onto a (typically unit) vector \f$\mathbf{n}\f$:
         * \f[
         * \mathrm{proj}_{\mathbf{n}}(\mathbf{v}) = (\mathbf{v}\cdot\mathbf{n})\mathbf{n}
         * \f]
         *
         * @param n Direction to project onto (typically unit length).
         * @return Projected vector.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector
        projected(const Vector& n) const noexcept;

        /**
         * @brief Casts components to another scalar type.
         *
         * @tparam To Destination scalar type.
         * @return Vector with converted component type.
         */
        template <typename To>
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<To, 4>
        cast_to() const noexcept;
    };

    /**
     * @brief Dot product of two 4D vectors.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    dot(const Vector<T, 4>& a, const Vector<T, 4>& b) noexcept;

    /**
     * @brief Reflects vector `v` about a normal direction in 4D.
     *
     * @param v Input direction.
     * @param normal Normal direction (typically unit length).
     * @return Reflected direction.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    reflected(const Vector<T, 4>& v, const Vector<T, 4>& normal) noexcept;

    /**
     * @brief Projects vector `v` onto direction `normal` in 4D.
     *
     * @param v Input vector.
     * @param normal Projection direction (typically unit length).
     * @return Projection of `v` onto `normal`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    projected(const Vector<T, 4>& v, const Vector<T, 4>& normal) noexcept;

    /** @brief Unary plus (returns a copy). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator+(const Vector<T, 4>& a);

    /** @brief Unary minus (negates all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator-(const Vector<T, 4>& a);

    /** @brief Vector + vector (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator+(const Vector<T, 4>& a, const Vector<T, 4>& b);

    /** @brief Vector - vector (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator-(const Vector<T, 4>& a, const Vector<T, 4>& b);

    /** @brief Scalar + vector (adds scalar to all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator+(T a, const Vector<T, 4>& b);

    /** @brief Vector + scalar (adds scalar to all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator+(const Vector<T, 4>& a, T b);

    /** @brief Scalar - vector (subtracts vector components from scalar). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator-(T a, const Vector<T, 4>& b);

    /** @brief Vector - scalar (subtracts scalar from all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator-(const Vector<T, 4>& a, T b);

    /** @brief Vector * scalar (scales all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator*(const Vector<T, 4>& a, T b);

    /** @brief Scalar * vector (scales all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator*(T a, const Vector<T, 4>& b);

    /** @brief Vector * vector (element-wise / Hadamard). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator*(const Vector<T, 4>& a, const Vector<T, 4>& b);

    /** @brief Vector / scalar (divides all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator/(const Vector<T, 4>& a, T b);

    /** @brief Scalar / vector (divides scalar by vector per component). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator/(T a, const Vector<T, 4>& b);

    /** @brief Vector / vector (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator/(const Vector<T, 4>& a, const Vector<T, 4>& b);

    /**
     * @brief Component-wise min of two vectors.
     *
     * @return `{min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w)}`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    min(const Vector<T, 4>& a, const Vector<T, 4>& b);

    /**
     * @brief Component-wise max of two vectors.
     *
     * @return `{max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w)}`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    max(const Vector<T, 4>& a, const Vector<T, 4>& b);

    /**
     * @brief Component-wise clamp.
     *
     * @details
     * \f[
     * \mathrm{clamp}(v,low,high)=\min(\max(v,low),high)
     * \f]
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    clamp(const Vector<T, 4>& v, const Vector<T, 4>& low, const Vector<T, 4>& high);

    /** @brief Component-wise ceil. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    ceil(const Vector<T, 4>& a);

    /** @brief Component-wise floor. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    floor(const Vector<T, 4>& a);

    /** @brief Component-wise absolute value. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    abs(const Vector<T, 4>& v);

    /** @brief Component-wise minimum (alias helper). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    cmin(const Vector<T, 4>& a, const Vector<T, 4>& b);

    /** @brief Component-wise maximum (alias helper). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    cmax(const Vector<T, 4>& a, const Vector<T, 4>& b);

} // namespace math

template <typename T>
using Vector4  = math::Vector<T, 4>;
using Vector4F = Vector4<float>;
using Vector4D = Vector4<double>;
using Point4UI = Vector4<std::uint32_t>;

} // namespace atlas

#include <atlas/math/vector/vector4.hpp>
