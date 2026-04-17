#pragma once

#include <atlas/math/vector/vector.h>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

namespace atlas {
namespace math {

    /**
     * @brief Specialized 2D vector with named components and 2D-specific operations.
     *
     * @details
     * This is a partial specialization of `Vector<T, N>` for `N = 2`, providing:
     * - Named fields `x` and `y` (in addition to index-based access).
     * - 2D-specific helpers such as scalar cross product, reflection/projection, and tangential vector.
     * - A compact representation suitable for host/device code paths.
     *
     * The vector participates in the expression-template system by inheriting from
     * `VectorExpression<T, Vector<T,2>>`, enabling efficient evaluation of composed
     * expressions.
     *
     * Storage semantics:
     * - `x` corresponds to index 0
     * - `y` corresponds to index 1
     * - `data()` returns a pointer to `x` (contiguous two-element storage)
     *
     * ---
     *
     * @tparam T Scalar type (must be trivially copyable).
     *
     * @note
     * - Equality operators perform exact element-wise comparisons.
     * - `normalize()` assumes non-zero length; behavior for zero-length vectors depends on
     *   the implementation in `vector2.hpp`.
     * - `major_axis()` / `minor_axis()` typically refer to the component with the largest/smallest
     *   absolute value (implementation-defined tie-breaking).
     */
    template <typename T>
    class Vector<T, 2> : public VectorExpression<T, Vector<T, 2>> {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    public:
        /// @brief X component (index 0).
        T x;
        /// @brief Y component (index 1).
        T y;

        /**
         * @brief Default constructor.
         *
         * @details
         * Initializes the vector to an implementation-defined state (commonly `{0,0}`).
         * See `vector2.hpp` for exact behavior.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Vector() noexcept;

        /// @brief Copy constructor.
        constexpr Vector(const Vector& v) noexcept = default;

        /**
         * @brief Constructs a vector with both components set to the same scalar.
         *
         * @param s Scalar assigned to both `x` and `y`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Vector(T s) noexcept;

        /**
         * @brief Constructs a vector from explicit components.
         *
         * @param x_ X component.
         * @param y_ Y component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Vector(T x_, T y_) noexcept;

        /**
         * @brief Constructs from an initializer list.
         *
         * @details
         * Expected order is `{x, y}`. If fewer than 2 values are provided, remaining
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
         * @brief Returns the dimension of the vector (always 2).
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        size() noexcept;

        /**
         * @brief Returns a pointer to contiguous storage (const).
         *
         * @return Pointer to `x`, followed by `y`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
        data() const noexcept;

        /**
         * @brief Returns a pointer to contiguous storage (mutable).
         *
         * @return Pointer to `x`, followed by `y`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
        data() noexcept;

        /**
         * @brief Element access (const) by index.
         *
         * @param i Index in `{0,1}` (`0 → x`, `1 → y`).
         * @return Const reference to the component.
         *
         * @note No bounds checking is implied.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        operator[](std::size_t i) const noexcept;

        /**
         * @brief Element access (mutable) by index.
         *
         * @param i Index in `{0,1}` (`0 → x`, `1 → y`).
         * @return Reference to the component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        operator[](std::size_t i) noexcept;

        /**
         * @brief Element access (const), intended as a safer accessor.
         *
         * @param i Index in `{0,1}`.
         * @return Const reference to the component.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        at(std::size_t i) const noexcept;

        /**
         * @brief Element access (mutable), intended as a safer accessor.
         *
         * @param i Index in `{0,1}`.
         * @return Reference to the component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        at(std::size_t i) noexcept;

        /**
         * @brief Sets both components to the same scalar.
         *
         * @param s Scalar to assign to `x` and `y`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set(T s) noexcept;

        /**
         * @brief Sets both components from two values.
         *
         * @tparam Args Must be exactly 2 arguments convertible to `T`.
         * @param args Values assigned in order: `x = args[0]`, `y = args[1]`.
         */
        template <typename... Args,
                  typename = std::enable_if_t<(sizeof...(Args) == 2)
                                              && (std::conjunction_v<std::is_convertible<Args, T>...>)>>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_values(Args... args) noexcept;

        /**
         * @brief Sets both components to zero.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_zero() noexcept;

        /** @brief Adds a scalar to both components in-place. */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(T v) noexcept;

        /** @brief Subtracts a scalar from both components in-place. */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        sub(T v) noexcept;

        /** @brief Multiplies both components by a scalar in-place. */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        mul(T v) noexcept;

        /**
         * @brief Divides both components by a scalar in-place.
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

        /**
         * @brief Multiplies by another vector element-wise in-place (Hadamard product).
         */
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
         * @brief Returns the smaller of the two component values.
         *
         * @return `min(x, y)`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        min() const noexcept;

        /**
         * @brief Returns the larger of the two component values.
         *
         * @return `max(x, y)`.
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
         * @brief Computes the dot product with another 2D vector.
         *
         * @details
         * \f[
         * (x_1,y_1)\cdot(x_2,y_2)=x_1x_2+y_1y_2
         * \f]
         *
         * @param v Other vector.
         * @return Dot product.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        dot(const Vector& v) const noexcept;

        /**
         * @brief Computes the 2D "cross product" as a scalar.
         *
         * @details
         * In 2D, the cross product is defined as the z-component of the 3D cross
         * product of `(x,y,0)` vectors:
         * \f[
         * \mathrm{cross}((x_1,y_1),(x_2,y_2)) = x_1y_2 - y_1x_2
         * \f]
         *
         * This value is useful for orientation tests and signed area computations.
         *
         * @param v Other vector.
         * @return Scalar cross product.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        cross(const Vector& v) const noexcept;

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
         * Typically returns the component index with the larger absolute value.
         *
         * @return 0 for X, 1 for Y.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
        major_axis() const noexcept;

        /**
         * @brief Returns the index of the minor axis.
         *
         * @details
         * Typically returns the component index with the smaller absolute value.
         *
         * @return 0 for X, 1 for Y.
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
         * @brief Returns the reflection of this vector about a unit normal.
         *
         * @details
         * Reflection formula:
         * \f[
         * \mathbf{r} = \mathbf{v} - 2(\mathbf{v}\cdot\mathbf{n})\mathbf{n}
         * \f]
         *
         * @param n Surface normal direction (typically unit length).
         * @return Reflected vector.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector
        reflected(const Vector& n) const noexcept;

        /**
         * @brief Projects this vector onto a direction `n`.
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
         * @brief Returns a perpendicular (tangential) vector.
         *
         * @details
         * A common 2D perpendicular is:
         * \f[
         * \mathbf{t} = (-y, x)
         * \f]
         *
         * This rotates the vector by +90 degrees (implementation-defined orientation).
         *
         * @return Perpendicular vector.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector
        tangential() const noexcept;

        /**
         * @brief Casts components to another scalar type.
         *
         * @tparam To Destination scalar type.
         * @return Vector with converted component type.
         */
        template <typename To>
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<To, 2>
        cast_to() const noexcept;
    };

    /**
     * @brief Dot product of two 2D vectors.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    dot(const Vector<T, 2>& a, const Vector<T, 2>& b) noexcept;

    /**
     * @brief Scalar 2D cross product of two 2D vectors.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    cross(const Vector<T, 2>& a, const Vector<T, 2>& b) noexcept;

    /**
     * @brief Reflects vector `v` about normal `normal`.
     *
     * @param v Input direction.
     * @param normal Surface normal (typically unit length).
     * @return Reflected direction.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    reflected(const Vector<T, 2>& v, const Vector<T, 2>& normal) noexcept;

    /**
     * @brief Projects vector `v` onto direction `normal`.
     *
     * @param v Input vector.
     * @param normal Projection direction (typically unit length).
     * @return Projection of `v` onto `normal`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    projected(const Vector<T, 2>& v, const Vector<T, 2>& normal) noexcept;

    /** @brief Unary plus (returns a copy). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator+(const Vector<T, 2>& a);

    /** @brief Unary minus (negates both components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator-(const Vector<T, 2>& a);

    /** @brief Vector + vector (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator+(const Vector<T, 2>& a, const Vector<T, 2>& b);

    /** @brief Vector - vector (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator-(const Vector<T, 2>& a, const Vector<T, 2>& b);

    /** @brief Scalar + vector (adds scalar to both components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator+(T a, const Vector<T, 2>& b);

    /** @brief Vector + scalar (adds scalar to both components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator+(const Vector<T, 2>& a, T b);

    /** @brief Scalar - vector (subtracts vector from scalar per component). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator-(T a, const Vector<T, 2>& b);

    /** @brief Vector - scalar (subtracts scalar from both components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator-(const Vector<T, 2>& a, T b);

    /** @brief Vector * scalar (scales both components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator*(const Vector<T, 2>& a, T b);

    /** @brief Scalar * vector (scales both components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator*(T a, const Vector<T, 2>& b);

    /** @brief Vector * vector (element-wise / Hadamard). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator*(const Vector<T, 2>& a, const Vector<T, 2>& b);

    /** @brief Vector / scalar (divides both components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator/(const Vector<T, 2>& a, T b);

    /** @brief Scalar / vector (divides scalar by vector per component). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator/(T a, const Vector<T, 2>& b);

    /** @brief Vector / vector (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator/(const Vector<T, 2>& a, const Vector<T, 2>& b);

    /**
     * @brief Component-wise min of two vectors.
     *
     * @return `{min(a.x,b.x), min(a.y,b.y)}`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    min(const Vector<T, 2>& a, const Vector<T, 2>& b);

    /**
     * @brief Component-wise max of two vectors.
     *
     * @return `{max(a.x,b.x), max(a.y,b.y)}`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    max(const Vector<T, 2>& a, const Vector<T, 2>& b);

    /**
     * @brief Component-wise clamp.
     *
     * @details
     * \f[
     * \mathrm{clamp}(v,low,high)=\min(\max(v,low),high)
     * \f]
     *
     * @return Clamped vector.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    clamp(const Vector<T, 2>& v, const Vector<T, 2>& low, const Vector<T, 2>& high);

    /** @brief Component-wise ceil. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    ceil(const Vector<T, 2>& a);

    /** @brief Component-wise floor. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    floor(const Vector<T, 2>& a);

    /** @brief Component-wise absolute value. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    abs(const Vector<T, 2>& v);

    /**
     * @brief Component-wise minimum (alias helper).
     *
     * @return `{min(a.x,b.x), min(a.y,b.y)}`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    cmin(const Vector<T, 2>& a, const Vector<T, 2>& b);

    /**
     * @brief Component-wise maximum (alias helper).
     *
     * @return `{max(a.x,b.x), max(a.y,b.y)}`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    cmax(const Vector<T, 2>& a, const Vector<T, 2>& b);

} // namespace math

template <typename T>
using Vector2  = math::Vector<T, 2>;
using Vector2F = Vector2<float>;
using Vector2D = Vector2<double>;
using Point2UI = Vector2<std::uint32_t>;

} // namespace atlas

#include <atlas/math/vector/vector2.hpp>
