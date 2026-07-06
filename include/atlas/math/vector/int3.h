/**
 * @file   int3.h
 * @brief  3D signed-integer vector type Int3 and related free functions.
 *
 * Int3 is a SIMD-independent vector type with x, y, z components,
 * designed to work on both CPU (Host) and CUDA GPU (Device). It is
 * typically used for grid/voxel indices and other integral coordinates,
 * and interoperates with Float3 via to_vector3() / to_vector3i().
 * All member and free functions explicitly declare their compilation
 * target (host/device) via ATLAS_ALL_DEVICE / ATLAS_HOST macros, and
 * force inlining via ATLAS_FORCE_INLINE to eliminate call overhead.
 */

#pragma once

#include <atlas/core/macros.h>       /**< ATLAS_ALL_DEVICE, ATLAS_HOST, ATLAS_FORCE_INLINE, ATLAS_NODISCARD, etc. */
#include <atlas/math/vector/bool3.h> /**< Bool3 type holding per-component comparison results */
#include <atlas/math/vector/float3.h> /**< Float3 conversion target/source for to_vector3() / to_vector3i() */

#include <cstddef> // std::size_t

namespace atlas {

/**
 * @class Int3
 * @brief 3D signed-integer vector.
 *
 * A general-purpose integral type representing 3-component index/coordinate
 * data such as grid cells or voxel positions. Since the member variables
 * x, y, z are laid out contiguously in memory, a pointer obtained via
 * `data()` can be passed directly to GPU kernels or C libraries.
 */
class Int3 {
public:
    int x; /**< First (X) component of the vector */
    int y; /**< Second (Y) component of the vector */
    int z; /**< Third (Z) component of the vector */

    /** @brief Default constructor. Initializes all components to 0. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Int3() noexcept
        : x(0)
        , y(0)
        , z(0) { }

    /** @brief Copy constructor (compiler-generated default). */
    constexpr Int3(const Int3&) noexcept = default;

    /**
     * @brief Scalar broadcast constructor. `Int3(2)` -> `{2, 2, 2}`.
     * @param s  Value assigned identically to all three components
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Int3(const int s) noexcept
        : x(s)
        , y(s)
        , z(s) { }

    /**
     * @brief Constructor specifying each component individually.
     * @param x_  X component
     * @param y_  Y component
     * @param z_  Z component
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Int3(const int x_, const int y_, const int z_) noexcept
        : x(x_)
        , y(y_)
        , z(z_) { }

    /** @brief Destructor (trivial). */
    ~Int3() noexcept = default;

    /** @brief Copy assignment operator (compiler-generated default). */
    Int3&
    operator=(const Int3&) noexcept = default;

    /**
     * @brief Returns the number of components in the vector.
     * @return Always 3
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size() noexcept {
        return 3;
    }

    /**
     * @brief Returns a const pointer to the first component (x).
     *
     * Since the three components are contiguous in memory, the
     * resulting pointer can be used as an array.
     * @return &x (const)
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const int*
    data() const noexcept {
        return &x;
    }

    /**
     * @brief Returns a mutable pointer to the first component (x).
     * @return &x (mutable)
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int*
    data() noexcept {
        return &x;
    }

    /**
     * @brief Accesses a component by index (read-only).
     * @param i  0=x, 1=y, 2=z
     * @return  Const reference to the i-th component
     * @warning No bounds checking. Undefined behavior if i >= 3.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const int&
    operator[](const std::size_t i) const noexcept {
        return (&x)[i];
    }

    /**
     * @brief Accesses a component by index (mutable).
     * @param i  0=x, 1=y, 2=z
     * @return  Reference to the i-th component
     * @warning No bounds checking. Undefined behavior if i >= 3.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int&
    operator[](const std::size_t i) noexcept {
        return (&x)[i];
    }

    /** @brief Sets all components to 0. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_zero() noexcept {
        x = y = z = 0;
    }

    /** @brief Vector addition assignment (component-wise). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3&
    operator+=(const Int3& v) noexcept {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    /** @brief Vector subtraction assignment (component-wise). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3&
    operator-=(const Int3& v) noexcept {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    /** @brief Scalar multiplication assignment. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3&
    operator*=(const int s) noexcept {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    /** @brief Checks whether two vectors are exactly equal. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Int3& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }

    /** @brief Checks whether two vectors are different. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Int3& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief Returns the smallest of the three components.
     * @return min(x, y, z)
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
    min() const noexcept {
        return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
    }

    /**
     * @brief Returns the largest of the three components.
     * @return max(x, y, z)
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
    max() const noexcept {
        return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
    }
};

/** @brief Vector + vector (component-wise). */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
operator+(const Int3& a, const Int3& b) noexcept {
    return Int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/** @brief Vector - vector (component-wise). */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
operator-(const Int3& a, const Int3& b) noexcept {
    return Int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/** @brief Unary minus. Negates the sign of each component. */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
operator-(const Int3& a) noexcept {
    return Int3(-a.x, -a.y, -a.z);
}

/** @brief Vector * scalar. */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
operator*(const Int3& a, const int s) noexcept {
    return Int3(a.x * s, a.y * s, a.z * s);
}

/** @brief Scalar * vector. */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
operator*(const int s, const Int3& a) noexcept {
    return a * s;
}

/** @brief Component-wise less-than (`<`) comparison. Returns a Bool3. */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator<(const Int3& a, const Int3& b) noexcept {
    return Bool3 { a.x < b.x, a.y < b.y, a.z < b.z };
}

/** @brief Component-wise less-than-or-equal (`<=`) comparison. */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator<=(const Int3& a, const Int3& b) noexcept {
    return Bool3 { a.x <= b.x, a.y <= b.y, a.z <= b.z };
}

/** @brief Component-wise greater-than (`>`) comparison. */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator>(const Int3& a, const Int3& b) noexcept {
    return Bool3 { a.x > b.x, a.y > b.y, a.z > b.z };
}

/** @brief Component-wise greater-than-or-equal (`>=`) comparison. */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator>=(const Int3& a, const Int3& b) noexcept {
    return Bool3 { a.x >= b.x, a.y >= b.y, a.z >= b.z };
}

/**
 * @brief Returns the component-wise minimum vector.
 * @return {min(a.x,b.x), min(a.y,b.y), min(a.z,b.z)}
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
min(const Int3& a, const Int3& b) noexcept {
    return Int3((a.x < b.x) ? a.x : b.x,
                (a.y < b.y) ? a.y : b.y,
                (a.z < b.z) ? a.z : b.z);
}

/**
 * @brief Returns the component-wise maximum vector.
 * @return {max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)}
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
max(const Int3& a, const Int3& b) noexcept {
    return Int3((a.x > b.x) ? a.x : b.x,
                (a.y > b.y) ? a.y : b.y,
                (a.z > b.z) ? a.z : b.z);
}

/**
 * @brief Clamps each component to the range [low, high].
 * @param v     Input vector
 * @param low   Lower bound vector
 * @param high  Upper bound vector
 * @return      Component-wise clamp(v, low, high)
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
clamp(const Int3& v, const Int3& low, const Int3& high) noexcept {
    return min(max(v, low), high);
}

/**
 * @brief Converts a Float3 to an Int3 by truncating each component.
 * @param v  Input floating-point vector
 * @return   {(int)v.x, (int)v.y, (int)v.z}
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int3
to_vector3i(const Float3& v) noexcept {
    return Int3(static_cast<int>(v.x),
                static_cast<int>(v.y),
                static_cast<int>(v.z));
}

/**
 * @brief Converts an Int3 to a Float3.
 * @param v  Input integer vector
 * @return   {(float)v.x, (float)v.y, (float)v.z}
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
to_vector3(const Int3& v) noexcept {
    return Float3(static_cast<float>(v.x),
                  static_cast<float>(v.y),
                  static_cast<float>(v.z));
}

} // namespace atlas
