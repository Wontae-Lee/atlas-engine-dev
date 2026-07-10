#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/vector/bool3.h>
#include <atlas/math/vector/float3.h>

#include <cstddef>

namespace atlas {

/**
 * @brief A three-component signed-integer vector, the engine's grid index type.
 *
 * The integer counterpart of Float3, with the same standard-layout guarantee
 * (&x is a length-3 int array). Used mainly for spatial-hash cell coordinates
 * and grid resolutions; it deliberately omits float-only operations such as
 * normalize and length. Trivially copyable and device-capturable.
 */
class Int3 {
public:
    int x; ///< First component.
    int y; ///< Second component.
    int z; ///< Third component.

    /**
     * @brief Construct the zero vector (0, 0, 0).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Int3() noexcept
        : x(0)
        , y(0)
        , z(0) { }

    /**
     * @brief Copy constructor; a plain component-wise copy.
     */
    constexpr Int3(const Int3&) noexcept = default;

    /**
     * @brief Construct a uniform vector with every component equal to @p s.
     *
     * Explicit so a bare int never implicitly converts to an Int3.
     *
     * @param s The value assigned to x, y and z.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Int3(const int s) noexcept
        : x(s)
        , y(s)
        , z(s) { }

    /**
     * @brief Construct from explicit components.
     *
     * @param x_ The x component.
     * @param y_ The y component.
     * @param z_ The z component.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Int3(const int x_, const int y_, const int z_) noexcept
        : x(x_)
        , y(y_)
        , z(z_) { }

    /**
     * @brief Trivial destructor.
     */
    ~Int3() noexcept = default;

    /**
     * @brief Copy assignment; a plain component-wise copy.
     */
    Int3&
    operator=(const Int3&) noexcept = default;

    /**
     * @brief The fixed component count.
     * @return Always 3.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size() noexcept {
        return 3;
    }

    /**
     * @brief Pointer to the first component for contiguous read access.
     * @return &x, addressing a length-3 int array.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const int*
    data() const noexcept {
        return &x;
    }

    /**
     * @brief Pointer to the first component for contiguous write access.
     * @return &x, addressing a length-3 int array.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int*
    data() noexcept {
        return &x;
    }

    /**
     * @brief Read a component by index, unchecked.
     * @param i Axis index; only 0, 1, 2 are valid. No bounds check.
     * @return Const reference to component @p i.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const int&
    operator[](const std::size_t i) const noexcept {
        return (&x)[i];
    }

    /**
     * @brief Access a component by index, unchecked.
     * @param i Axis index; only 0, 1, 2 are valid. No bounds check.
     * @return Mutable reference to component @p i.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int&
    operator[](const std::size_t i) noexcept {
        return (&x)[i];
    }

    /**
     * @brief Reset the vector to (0, 0, 0).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_zero() noexcept {
        x = y = z = 0;
    }

    /**
     * @brief Component-wise add a vector and return *this.
     * @param v The vector addend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3&
    operator+=(const Int3& v) noexcept {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    /**
     * @brief Component-wise subtract a vector and return *this.
     * @param v The vector subtrahend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3&
    operator-=(const Int3& v) noexcept {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    /**
     * @brief Scale every component by an integer and return *this.
     * @param s The multiplier.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3&
    operator*=(const int s) noexcept {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    /**
     * @brief Exact component-wise equality.
     * @param other The vector to compare against.
     * @return True when all three components are equal.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Int3& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }

    /**
     * @brief Negation of operator==.
     * @param other The vector to compare against.
     * @return True when any component differs.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Int3& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief The smallest of the three components.
     * @return min(x, y, z).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
    min() const noexcept {
        return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
    }

    /**
     * @brief The largest of the three components.
     * @return max(x, y, z).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
    max() const noexcept {
        return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
    }
};

/**
 * @brief Component-wise integer vector addition.
 * @param a Left operand.
 * @param b Right operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
operator+(const Int3& a, const Int3& b) noexcept {
    return Int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * @brief Component-wise integer vector subtraction.
 * @param a Left operand.
 * @param b Right operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
operator-(const Int3& a, const Int3& b) noexcept {
    return Int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * @brief Unary negation.
 * @param a The operand.
 * @return (-a.x, -a.y, -a.z).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
operator-(const Int3& a) noexcept {
    return Int3(-a.x, -a.y, -a.z);
}

/**
 * @brief Vector times integer scalar.
 * @param a The vector.
 * @param s The multiplier.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
operator*(const Int3& a, const int s) noexcept {
    return Int3(a.x * s, a.y * s, a.z * s);
}

/**
 * @brief Integer scalar times vector (commutes with the overload above).
 * @param s The multiplier.
 * @param a The vector.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
operator*(const int s, const Int3& a) noexcept {
    return a * s;
}

/**
 * @brief Per-axis "less than", returning a boolean triple.
 * @param a Left operand.
 * @param b Right operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator<(const Int3& a, const Int3& b) noexcept {
    return Bool3 { a.x < b.x, a.y < b.y, a.z < b.z };
}

/**
 * @brief Per-axis "less than or equal", returning a boolean triple.
 * @param a Left operand.
 * @param b Right operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator<=(const Int3& a, const Int3& b) noexcept {
    return Bool3 { a.x <= b.x, a.y <= b.y, a.z <= b.z };
}

/**
 * @brief Per-axis "greater than", returning a boolean triple.
 * @param a Left operand.
 * @param b Right operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator>(const Int3& a, const Int3& b) noexcept {
    return Bool3 { a.x > b.x, a.y > b.y, a.z > b.z };
}

/**
 * @brief Per-axis "greater than or equal", returning a boolean triple.
 * @param a Left operand.
 * @param b Right operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator>=(const Int3& a, const Int3& b) noexcept {
    return Bool3 { a.x >= b.x, a.y >= b.y, a.z >= b.z };
}

/**
 * @brief Component-wise minimum of two integer vectors.
 * @param a First operand.
 * @param b Second operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
min(const Int3& a, const Int3& b) noexcept {
    return Int3((a.x < b.x) ? a.x : b.x,
                (a.y < b.y) ? a.y : b.y,
                (a.z < b.z) ? a.z : b.z);
}

/**
 * @brief Component-wise maximum of two integer vectors.
 * @param a First operand.
 * @param b Second operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
max(const Int3& a, const Int3& b) noexcept {
    return Int3((a.x > b.x) ? a.x : b.x,
                (a.y > b.y) ? a.y : b.y,
                (a.z > b.z) ? a.z : b.z);
}

/**
 * @brief Clamp each component of @p v into the box [low, high].
 * @param v The vector to clamp.
 * @param low Per-axis lower bounds.
 * @param high Per-axis upper bounds.
 * @return min(max(v, low), high). Assumes low <= high on every axis.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
clamp(const Int3& v, const Int3& low, const Int3& high) noexcept {
    return min(max(v, low), high);
}

/**
 * @brief Convert a Float3 to an Int3 by truncation toward zero.
 *
 * Each component is cast with static_cast<int>, which truncates (rounds toward
 * zero), not floors. For negative coordinates this differs from floor(); callers
 * mapping positions to grid cells must account for that on the negative side.
 *
 * @param v The float vector to convert.
 * @return The truncated integer vector.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
to_vector3i(const Float3& v) noexcept {
    return Int3(static_cast<int>(v.x),
                static_cast<int>(v.y),
                static_cast<int>(v.z));
}

/**
 * @brief Convert an Int3 to a Float3 by widening each component.
 * @param v The integer vector to convert.
 * @return The float vector.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
to_vector3(const Int3& v) noexcept {
    return Float3(static_cast<float>(v.x),
                  static_cast<float>(v.y),
                  static_cast<float>(v.z));
}

}