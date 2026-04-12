#pragma once

#include <atlas/math/vector/vector.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace atlas {
namespace math {

    /**
     * @brief Specialized 3D vector with named components and 3D-specific operations.
     *
     * @details
     * This is a partial specialization of `Vector<T, N>` for `N = 3`, providing:
     * - Named fields `x`, `y`, `z` (alongside index-based access).
     * - 3D-specific operations: vector cross product, generation of tangential basis vectors,
     *   reflection/projection helpers, and axis queries.
     * - A compact POD-like layout suitable for host and device execution.
     *
     * The type participates in Atlas' expression-template system by inheriting from
     * `VectorExpression<T, Vector<T,3>>`, enabling efficient composition of expressions.
     *
     * Storage semantics:
     * - `x` corresponds to index 0
     * - `y` corresponds to index 1
     * - `z` corresponds to index 2
     * - `data()` returns a pointer to `x` (contiguous three-element storage)
     *
     * ---
     *
     * @tparam T Scalar type (must be trivially copyable).
     *
     * @note
     * - Equality operators perform exact element-wise comparisons.
     * - `normalize()` assumes non-zero length; behavior for zero-length vectors depends on
     *   the implementation in `vector3.hpp`.
     * - `tangential()` / `tangential(normal)` typically return two orthonormal vectors that span
     *   the plane perpendicular to `normal` (implementation-defined tie-breaking and handedness).
     */
    template <typename T>
    class Vector<T, 3> : public VectorExpression<T, Vector<T, 3>> {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    public:
        /// @brief X component (index 0).
        T x;
        /// @brief Y component (index 1).
        T y;
        /// @brief Z component (index 2).
        T z;

        /**
         * @brief Default constructor.
         *
         * @details
         * Initializes the vector to an implementation-defined state (commonly `{0,0,0}`).
         * See `vector3.hpp` for exact behavior.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Vector() noexcept;

        /// @brief Copy constructor.
        constexpr Vector(const Vector& v) noexcept = default;

        /**
         * @brief Constructs a vector with all components set to the same scalar.
         *
         * @param s Scalar assigned to `x`, `y`, and `z`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Vector(T s) noexcept;

        /**
         * @brief Constructs a vector from explicit components.
         *
         * @param x_ X component.
         * @param y_ Y component.
         * @param z_ Z component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Vector(T x_, T y_, T z_) noexcept;

        /**
         * @brief Constructs from an initializer list.
         *
         * @details
         * Expected order is `{x, y, z}`. If fewer than 3 values are provided, remaining
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
         * @brief Returns the dimension of the vector (always 3).
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        size() noexcept;

        /**
         * @brief Returns a pointer to contiguous storage (const).
         *
         * @return Pointer to `x`, followed by `y`, then `z`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
        data() const noexcept;

        /**
         * @brief Returns a pointer to contiguous storage (mutable).
         *
         * @return Pointer to `x`, followed by `y`, then `z`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
        data() noexcept;

        /**
         * @brief Element access (const) by index.
         *
         * @param i Index in `{0,1,2}` (`0 → x`, `1 → y`, `2 → z`).
         * @return Const reference to the component.
         *
         * @note No bounds checking is implied.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        operator[](std::size_t i) const noexcept;

        /**
         * @brief Element access (mutable) by index.
         *
         * @param i Index in `{0,1,2}` (`0 → x`, `1 → y`, `2 → z`).
         * @return Reference to the component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        operator[](std::size_t i) noexcept;

        /**
         * @brief Element access (const), intended as a safer accessor.
         *
         * @param i Index in `{0,1,2}`.
         * @return Const reference to the component.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        at(std::size_t i) const noexcept;

        /**
         * @brief Element access (mutable), intended as a safer accessor.
         *
         * @param i Index in `{0,1,2}`.
         * @return Reference to the component.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        at(std::size_t i) noexcept;

        /// @brief Copy assignment.
        Vector&
        operator=(const Vector& rhs) noexcept = default;

        /**
         * @brief Sets all components to the same scalar.
         *
         * @param s Scalar to assign to `x`, `y`, and `z`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set(T s) noexcept;

        /**
         * @brief Sets all components from three values.
         *
         * @tparam Args Must be exactly 3 arguments convertible to `T`.
         * @param args Values assigned in order: `x`, `y`, `z`.
         */
        template <typename... Args,
                  typename = std::enable_if_t<(sizeof...(Args) == 3)
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
         * @brief Returns the minimum of the three component values.
         *
         * @return `min(x, y, z)`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        min() const noexcept;

        /**
         * @brief Returns the maximum of the three component values.
         *
         * @return `max(x, y, z)`.
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
         * @brief Computes the dot product with another 3D vector.
         *
         * @details
         * \f[
         * (x_1,y_1,z_1)\cdot(x_2,y_2,z_2)=x_1x_2+y_1y_2+z_1z_2
         * \f]
         *
         * @param v Other vector.
         * @return Dot product.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        dot(const Vector& v) const noexcept;

        /**
         * @brief Computes the 3D cross product with another vector.
         *
         * @details
         * \f[
         * \mathbf{a}\times\mathbf{b} =
         * \begin{pmatrix}
         * a_y b_z - a_z b_y \\
         * a_z b_x - a_x b_z \\
         * a_x b_y - a_y b_x
         * \end{pmatrix}
         * \f]
         *
         * @param v Other vector.
         * @return Cross product vector.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector
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
         * Typically returns the component index with the largest absolute value
         * (implementation-defined tie-breaking).
         *
         * @return Index in `{0,1,2}`.
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
         * @return Index in `{0,1,2}`.
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
         * @brief Returns two tangential vectors orthogonal to this vector (treated as a normal).
         *
         * @details
         * Produces a pair `(t0, t1)` such that:
         * - `t0 ⟂ normal`
         * - `t1 ⟂ normal`
         * - `t0 ⟂ t1`
         *
         * Typically used to construct a local coordinate frame on a surface from a normal.
         * Handedness and tie-breaking are implementation-defined.
         *
         * @return Tuple `{t0, t1}` forming a tangent basis.
         *
         * @note If the vector is zero-length, the result is implementation-defined.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::tuple<Vector, Vector>
        tangential() const noexcept;

        /**
         * @brief Casts components to another scalar type.
         *
         * @tparam To Destination scalar type.
         * @return Vector with converted component type.
         */
        template <typename To>
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<To, 3>
        cast_to() const noexcept;
    };

    /**
     * @brief Cross product of two 3D vectors.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    cross(const Vector<T, 3>& a, const Vector<T, 3>& b) noexcept;

    /**
     * @brief Dot product of two 3D vectors.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    dot(const Vector<T, 3>& a, const Vector<T, 3>& b) noexcept;

    /**
     * @brief Reflects vector `v` about a normal direction.
     *
     * @param v Input direction.
     * @param normal Surface normal (typically unit length).
     * @return Reflected direction.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    reflected(const Vector<T, 3>& v, const Vector<T, 3>& normal) noexcept;

    /**
     * @brief Projects vector `v` onto direction `normal`.
     *
     * @param v Input vector.
     * @param normal Projection direction (typically unit length).
     * @return Projection of `v` onto `normal`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    projected(const Vector<T, 3>& v, const Vector<T, 3>& normal) noexcept;

    /**
     * @brief Builds a tangent basis from a normal direction.
     *
     * @details
     * Returns two vectors `{t0, t1}` spanning the plane orthogonal to `normal`.
     * Typically used for shading frames and local parameterizations.
     *
     * @param normal Normal direction (ideally unit length).
     * @return Tuple `{t0, t1}` tangent basis.
     *
     * @note Behavior for zero-length normals is implementation-defined.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::tuple<Vector<T, 3>, Vector<T, 3>>
    tangential(const Vector<T, 3>& normal) noexcept;

    /** @brief Unary plus (returns a copy). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator+(const Vector<T, 3>& a);

    /** @brief Scalar + vector (adds scalar to all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator+(T a, const Vector<T, 3>& b);

    /** @brief Vector + scalar (adds scalar to all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator+(const Vector<T, 3>& a, T b);

    /** @brief Vector + vector (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator+(const Vector<T, 3>& a, const Vector<T, 3>& b);

    /** @brief Unary minus (negates all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator-(const Vector<T, 3>& a);

    /** @brief Vector - scalar (subtracts scalar from all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator-(const Vector<T, 3>& a, T b);

    /** @brief Scalar - vector (subtracts vector components from scalar). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator-(T a, const Vector<T, 3>& b);

    /** @brief Vector - vector (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator-(const Vector<T, 3>& a, const Vector<T, 3>& b);

    /** @brief Vector * scalar (scales all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator*(const Vector<T, 3>& a, T b);

    /** @brief Scalar * vector (scales all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator*(T a, const Vector<T, 3>& b);

    /** @brief Vector * vector (element-wise / Hadamard). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator*(const Vector<T, 3>& a, const Vector<T, 3>& b);

    /** @brief Vector / scalar (divides all components). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator/(const Vector<T, 3>& a, T b);

    /** @brief Scalar / vector (divides scalar by vector per component). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator/(T a, const Vector<T, 3>& b);

    /** @brief Vector / vector (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator/(const Vector<T, 3>& a, const Vector<T, 3>& b);

    /**
     * @brief Component-wise min of two vectors.
     *
     * @return `{min(a.x,b.x), min(a.y,b.y), min(a.z,b.z)}`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    min(const Vector<T, 3>& a, const Vector<T, 3>& b);

    /**
     * @brief Component-wise max of two vectors.
     *
     * @return `{max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)}`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    max(const Vector<T, 3>& a, const Vector<T, 3>& b);

    /**
     * @brief Component-wise clamp.
     *
     * @details
     * \f[
     * \mathrm{clamp}(v,low,high)=\min(\max(v,low),high)
     * \f]
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    clamp(const Vector<T, 3>& v, const Vector<T, 3>& low, const Vector<T, 3>& high);

    /** @brief Component-wise ceil. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    ceil(const Vector<T, 3>& a);

    /** @brief Component-wise floor. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    floor(const Vector<T, 3>& a);

    /** @brief Component-wise absolute value. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    abs(const Vector<T, 3>& v);

    /** @brief Component-wise minimum (alias helper). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    cmin(const Vector<T, 3>& a, const Vector<T, 3>& b);

    /** @brief Component-wise maximum (alias helper). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    cmax(const Vector<T, 3>& a, const Vector<T, 3>& b);

    /**
     * @brief Cast helper for 3D vectors.
     *
     * @tparam To Destination scalar type.
     * @tparam From Source scalar type.
     * @param v Input vector.
     * @return Vector with components converted to `To`.
     */
    template <typename To, typename From>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<To, 3>
    cast_to(const Vector<From, 3>& v) noexcept;

} // namespace math

template <typename T>
using Vector3  = math::Vector<T, 3>;
using Vector3F = Vector3<float>;
using Vector3D = Vector3<double>;
using Point3UI = Vector3<std::uint32_t>;

} // namespace atlas

#include <atlas/math/vector/vector3.hpp>
