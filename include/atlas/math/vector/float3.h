/// @file   float3.h
/// @brief  3D single-precision floating point vector type Float3 and related free functions.
///
/// Float3 is a SIMD-independent vector type with x, y, z components,
/// designed to work on both CPU (Host) and CUDA GPU (Device).
/// All member and free functions explicitly declare their compilation
/// target (host/device) via ATLAS_ALL_DEVICE / ATLAS_HOST macros, and
/// force inlining via ATLAS_FORCE_INLINE to eliminate call overhead.
///
/// ### Design principles
/// - **Value type**: holds three floats directly, not pointers/references.
/// - **noexcept guarantee**: no operation throws exceptions.
/// - **FMA usage**: dot / length_squared and other accumulation operations
///   use `std::fma` to minimize rounding error.
///
/// @note In CUDA device code, standard math functions such as `std::fma`
///       and `std::sqrt` are replaced by NVCC's `__device__` overloads.

#pragma once

#include <atlas/core/macros.h>        ///< ATLAS_ALL_DEVICE, ATLAS_HOST, ATLAS_FORCE_INLINE, ATLAS_NODISCARD, etc.
#include <atlas/math/constants.h>     ///< Project-specific math utilities such as sqrt_nonnegative()
#include <atlas/math/vector/bool3.h>  ///< Bool3 type holding per-component comparison results

#include <algorithm>       // std::min / std::max (fallback on some platforms)
#include <cmath>           // std::sqrt, std::fma, std::abs, std::ceil, std::floor, etc.
#include <cstddef>         // std::size_t
#include <initializer_list>
#include <tuple>           // std::tuple (return type of tangential())

namespace atlas {

/// @class Float3
/// @brief 3D single-precision floating point vector.
///
/// A general-purpose math type representing 3-component float data such
/// as spatial positions, directions, normal vectors, or colors. Since
/// the member variables x, y, z are laid out contiguously in memory, a
/// pointer obtained via `data()` can be passed directly to GPU kernels
/// or C libraries.
///
/// @code{.cpp}
/// atlas::Float3 n(0.f, 1.f, 0.f);          // upward normal
/// atlas::Float3 v(1.f, 2.f, 3.f);
/// float d   = v.dot(n);                     // dot product
/// auto  ref = v.reflected(n);               // reflected vector
/// @endcode
class Float3 {
public:
    float x; ///< First (X) component of the vector
    float y; ///< Second (Y) component of the vector
    float z; ///< Third (Z) component of the vector

    // -------------------------------------------------------------------------
    // Constructors / Destructor
    // -------------------------------------------------------------------------

    /// @brief Default constructor. Initializes all components to 0.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Float3() noexcept
        : x(0.0f)
        , y(0.0f)
        , z(0.0f) { }

    /// @brief Copy constructor (compiler-generated default).
    constexpr Float3(const Float3&) noexcept = default;

    /// @brief Scalar broadcast constructor. `Float3(2.f)` -> `{2, 2, 2}`.
    /// @param s  Value assigned identically to all three components
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Float3(const float s) noexcept
        : x(s)
        , y(s)
        , z(s) { }

    /// @brief Constructor specifying each component individually.
    /// @param x_  X component
    /// @param y_  Y component
    /// @param z_  Z component
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Float3(const float x_, const float y_, const float z_) noexcept
        : x(x_)
        , y(y_)
        , z(z_) { }

    /// @brief Initializer-list constructor (host only).
    ///
    /// Supports brace initialization such as `Float3{1.f, 2.f, 3.f}`.
    /// If fewer than 3 elements are given, the missing components are
    /// filled with 0.
    ///
    /// @param list  Initializer list of float values (1 to 3 elements)
    /// @note  Not available in CUDA device code (ATLAS_HOST only).
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Float3(std::initializer_list<float> list) noexcept {
        const float* it = list.begin();
        x               = (it != list.end()) ? *it++ : 0.0f;
        y               = (it != list.end()) ? *it++ : 0.0f;
        z               = (it != list.end()) ? *it++ : 0.0f;
    }

    /// @brief Destructor (trivial).
    ~Float3() noexcept = default;

    /// @brief Copy assignment operator (compiler-generated default).
    Float3& operator=(const Float3&) noexcept = default;

    // -------------------------------------------------------------------------
    // Size / Data access
    // -------------------------------------------------------------------------

    /// @brief Returns the number of components in the vector.
    /// @return Always 3
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size() noexcept {
        return 3;
    }

    /// @brief Returns a const pointer to the first component (x).
    ///
    /// Since the three components are contiguous in memory, the
    /// resulting pointer can be used as an array.
    /// @return &x (const)
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float*
    data() const noexcept {
        return &x;
    }

    /// @brief Returns a mutable pointer to the first component (x).
    /// @return &x (mutable)
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float*
    data() noexcept {
        return &x;
    }

    /// @brief Accesses a component by index (read-only).
    /// @param i  0=x, 1=y, 2=z
    /// @return  Const reference to the i-th component
    /// @warning No bounds checking. Undefined behavior if i >= 3.
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    operator[](const std::size_t i) const noexcept {
        return (&x)[i];
    }

    /// @brief Accesses a component by index (mutable).
    /// @param i  0=x, 1=y, 2=z
    /// @return  Reference to the i-th component
    /// @warning No bounds checking. Undefined behavior if i >= 3.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    operator[](const std::size_t i) noexcept {
        return (&x)[i];
    }

    /// @brief Same unchecked accessor as operator[] (const).
    /// @param i  0=x, 1=y, 2=z
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    at(const std::size_t i) const noexcept {
        return (&x)[i];
    }

    /// @brief Same unchecked accessor as operator[] (mutable).
    /// @param i  0=x, 1=y, 2=z
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    at(const std::size_t i) noexcept {
        return (&x)[i];
    }

    // -------------------------------------------------------------------------
    // Value assignment
    // -------------------------------------------------------------------------

    /// @brief Sets all components to the same scalar value.
    /// @param s  Value to assign
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set(const float s) noexcept {
        x = y = z = s;
    }

    /// @brief Sets the three components to individual values.
    /// @param x_  New X component
    /// @param y_  New Y component
    /// @param z_  New Z component
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_values(const float x_, const float y_, const float z_) noexcept {
        x = x_;
        y = y_;
        z = z_;
    }

    /// @brief Sets all components to 0.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_zero() noexcept {
        x = y = z = 0.0f;
    }

    // -------------------------------------------------------------------------
    // Scalar arithmetic (in-place)
    // -------------------------------------------------------------------------

    /// @brief Adds a scalar to all components. Equivalent to `*this += v`.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const float v) noexcept {
        x += v; y += v; z += v;
    }

    /// @brief Subtracts a scalar from all components.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const float v) noexcept {
        x -= v; y -= v; z -= v;
    }

    /// @brief Multiplies all components by a scalar.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    mul(const float v) noexcept {
        x *= v; y *= v; z *= v;
    }

    /// @brief Divides all components by a scalar.
    ///
    /// Pre-computes 1/v and applies it as a multiplication, reducing the
    /// number of divisions.
    /// @param v  If 0, the result becomes infinity (inf); callers must check.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    div(const float v) noexcept {
        const float inv = 1.0f / v; // Pre-compute reciprocal -> handled with 3 multiplications
        x *= inv; y *= inv; z *= inv;
    }

    // -------------------------------------------------------------------------
    // Vector arithmetic (in-place, component-wise)
    // -------------------------------------------------------------------------

    /// @brief Component-wise vector addition. x+=v.x, y+=v.y, z+=v.z.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const Float3& v) noexcept {
        x += v.x; y += v.y; z += v.z;
    }

    /// @brief Component-wise vector subtraction.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const Float3& v) noexcept {
        x -= v.x; y -= v.y; z -= v.z;
    }

    /// @brief Component-wise vector multiplication (Hadamard product).
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    mul(const Float3& v) noexcept {
        x *= v.x; y *= v.y; z *= v.z;
    }

    /// @brief Component-wise vector division.
    /// @warning Produces inf/NaN if any component of v is 0.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    div(const Float3& v) noexcept {
        x /= v.x; y /= v.y; z /= v.z;
    }

    // -------------------------------------------------------------------------
    // Component min / max
    // -------------------------------------------------------------------------

    /// @brief Returns the smallest of the three components.
    /// @return min(x, y, z)
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    min() const noexcept {
        // Handles the 3-way comparison with two nested ternary operators (minimizes branching)
        return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
    }

    /// @brief Returns the largest of the three components.
    /// @return max(x, y, z)
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    max() const noexcept {
        return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
    }

    // -------------------------------------------------------------------------
    // Compound assignment operators
    // -------------------------------------------------------------------------

    /// @brief Scalar addition assignment.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3& operator+=(const float v) noexcept { add(v); return *this; }
    /// @brief Scalar subtraction assignment.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3& operator-=(const float v) noexcept { sub(v); return *this; }
    /// @brief Scalar multiplication assignment.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3& operator*=(const float v) noexcept { mul(v); return *this; }
    /// @brief Scalar division assignment.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3& operator/=(const float v) noexcept { div(v); return *this; }

    /// @brief Vector addition assignment (component-wise).
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3& operator+=(const Float3& v) noexcept { add(v); return *this; }
    /// @brief Vector subtraction assignment (component-wise).
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3& operator-=(const Float3& v) noexcept { sub(v); return *this; }
    /// @brief Vector multiplication assignment (component-wise Hadamard).
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3& operator*=(const Float3& v) noexcept { mul(v); return *this; }
    /// @brief Vector division assignment (component-wise).
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3& operator/=(const Float3& v) noexcept { div(v); return *this; }

    // -------------------------------------------------------------------------
    // Equality comparison
    // -------------------------------------------------------------------------

    /// @brief Checks whether two vectors are exactly equal (bit-exact).
    ///
    /// @warning For results of floating-point computation, an
    ///          epsilon-based comparison is safer.
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Float3& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }

    /// @brief Checks whether two vectors are different.
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Float3& other) const noexcept {
        return !(*this == other);
    }

    // -------------------------------------------------------------------------
    // Vector algebra
    // -------------------------------------------------------------------------

    /// @brief Computes the dot product.
    ///
    /// `dot(a, b) = a.x*b.x + a.y*b.y + a.z*b.z`
    ///
    /// Implemented with two FMA (Fused Multiply-Add) calls to reduce
    /// rounding error:
    ///   `fma(z, v.z, fma(y, v.y, x*v.x))`
    ///
    /// @param v  Vector to take the dot product with
    /// @return   Scalar dot product value
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    dot(const Float3& v) const noexcept {
        return std::fma(z, v.z, std::fma(y, v.y, x * v.x));
    }

    /// @brief Computes the cross product.
    ///
    /// The resulting vector is perpendicular to both *this and v,
    /// following a right-handed coordinate system.
    ///
    /// @param v  Vector to take the cross product with
    /// @return   *this x v
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    cross(const Float3& v) const noexcept {
        return Float3(y * v.z - z * v.y,   // cx = ay*bz - az*by
                      z * v.x - x * v.z,   // cy = az*bx - ax*bz
                      x * v.y - y * v.x);  // cz = ax*by - ay*bx
    }

    /// @brief Returns the squared length of the vector.
    ///
    /// Computed without calling sqrt, so it is faster than length()
    /// when only comparing distances.
    /// Uses two FMA calls: `fma(z, z, fma(y, y, x*x))`
    ///
    /// @return |*this|^2
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length_squared() const noexcept {
        return std::fma(z, z, std::fma(y, y, x * x));
    }

    /// @brief Returns the Euclidean length (L2 norm) of the vector.
    /// @return |*this| = sqrt(x^2 + y^2 + z^2)
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length() const noexcept {
        return std::sqrt(length_squared());
    }

    /// @brief Returns the index of the component with the largest value.
    ///
    /// Commonly used for selecting a split axis in a BVH (Bounding
    /// Volume Hierarchy), for example.
    ///
    /// @return Index (0=x, 1=y, 2=z) of the largest component
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    major_axis() const noexcept {
        if (x >= y && x >= z) return 0; // x is the maximum
        if (y >= x && y >= z) return 1; // y is the maximum
        return 2;                       // z is the maximum
    }

    /// @brief Returns the index of the component with the smallest value.
    ///
    /// Used, for example, when selecting a stable axis for tangential
    /// basis computation.
    ///
    /// @return Index (0=x, 1=y, 2=z) of the smallest component
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    minor_axis() const noexcept {
        if (x <= y && x <= z) return 0;
        if (y <= x && y <= z) return 1;
        return 2;
    }

    // -------------------------------------------------------------------------
    // Normalization
    // -------------------------------------------------------------------------

    /// @brief Normalizes the vector to unit length in place.
    ///
    /// If the length is 0, does nothing (vector is preserved).
    /// Multiplies by the inverse square root instead of dividing,
    /// replacing a division with a multiplication.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    normalize() noexcept {
        const float ls = length_squared();
        if (ls == 0.0f) return; // Guard against zero vector: cannot normalize
        const float inv = 1.0f / std::sqrt(ls); // Compute reciprocal only once
        x *= inv; y *= inv; z *= inv;
    }

    /// @brief Returns a new normalized vector without modifying *this.
    ///
    /// If the vector is zero, returns the original unchanged.
    ///
    /// @return *this if |*this| == 0, otherwise *this / |*this|
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    normalized() const noexcept {
        const float ls = length_squared();
        if (ls == 0.0f) return *this; // Zero vector: return the original
        const float inv = 1.0f / std::sqrt(ls);
        return Float3(x * inv, y * inv, z * inv);
    }

    // -------------------------------------------------------------------------
    // Reflection / Projection
    // -------------------------------------------------------------------------

    /// @brief Returns the reflection of this vector about a unit normal n.
    ///
    /// Reflection formula:  r = v - 2*(v.n)*n
    ///
    /// @param n  Unit normal vector (assumed to have length 1)
    /// @return   Reflected vector
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    reflected(const Float3& n) const noexcept {
        const float d = dot(n); // Magnitude of the component along the normal
        return Float3(x - 2.0f * d * n.x,
                      y - 2.0f * d * n.y,
                      z - 2.0f * d * n.z);
    }

    /// @brief Returns the projection (tangential component) onto the
    ///        plane perpendicular to normal n.
    ///
    /// Projection formula:  proj = v - (v.n)*n
    ///
    /// Removes the normal component, leaving only the tangential
    /// component. Synonymous with `rejected`.
    ///
    /// @param n  Unit normal vector
    /// @return   Tangential component perpendicular to n
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    projected(const Float3& n) const noexcept {
        const float d = dot(n); // Scalar projection along the normal
        return Float3(x - d * n.x,
                      y - d * n.y,
                      z - d * n.z);
    }

    // -------------------------------------------------------------------------
    // Tangential basis construction
    // -------------------------------------------------------------------------

    /// @brief Generates two tangent vectors (t1, t2) with *this as the normal.
    ///
    /// Uses a stable Frisvad/Hughes-Möller style algorithm:
    /// - If |x| > |y|, construct t1 in the xz plane.
    /// - Otherwise, construct t1 in the yz plane.
    /// - t2 = (*this x t1).normalized()
    ///
    /// Forms a proper orthonormal basis (ONB) when *this is a unit vector.
    ///
    /// @return {t1, t2}  Tuple of the two tangent vectors
    /// @note  Guards against the degenerate case d2 == 0 by adding 1 when
    ///        computing `inv`.
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::tuple<Float3, Float3>
    tangential() const noexcept {
        const float ax = std::abs(x);
        const float ay = std::abs(y);
        Float3 t1;

        if (ax > ay) {
            // If the x component is larger, compute t1 based on the xz plane
            // t1 = normalize((-z, 0, x)): remove the y component, then normalize
            const float d2  = x * x + z * z;
            // Guard against degeneracy (d2 == 0): add 1 to the denominator so it never hits zero
            const float inv = 1.0f / std::sqrt(d2 + (d2 == 0.0f ? 1.0f : 0.0f));
            t1              = Float3(-z * inv, 0.0f, x * inv);
        } else {
            // If the y component is larger or equal, compute t1 based on the yz plane
            // t1 = normalize((0, z, -y)): remove the x component, then normalize
            const float d2  = y * y + z * z;
            const float inv = 1.0f / std::sqrt(d2 + (d2 == 0.0f ? 1.0f : 0.0f));
            t1              = Float3(0.0f, z * inv, -y * inv);
        }

        // t2 = normal x t1, re-normalized to prevent accumulated numerical error
        const Float3 t2 = cross(t1).normalized();
        return { t1, t2 };
    }
};

// =============================================================================
// Free functions
// =============================================================================
// Semantically identical to the member functions, but provide the
// `dot(a, b)` call-style syntax. Automatically discovered in the
// `atlas::` namespace via ADL (Argument-Dependent Lookup).

/// @brief Computes the cross product of two vectors.
/// @param a  Operand vector A
/// @param b  Operand vector B
/// @return   a x b
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
cross(const Float3& a, const Float3& b) noexcept {
    return Float3(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

/// @brief Computes the dot product of two vectors. Uses two FMA calls.
/// @param a  Operand vector A
/// @param b  Operand vector B
/// @return   a . b
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
dot(const Float3& a, const Float3& b) noexcept {
    return std::fma(a.z, b.z, std::fma(a.y, b.y, a.x * b.x));
}

/// @brief Returns the reflection about a unit normal.
/// @param v       Incident vector
/// @param normal  Unit normal vector (assumes |normal| = 1)
/// @return        v - 2*(v.normal)*normal
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
reflected(const Float3& v, const Float3& normal) noexcept {
    const float d = dot(v, normal);
    return Float3(v.x - 2.0f * d * normal.x,
                  v.y - 2.0f * d * normal.y,
                  v.z - 2.0f * d * normal.z);
}

/// @brief Returns the projection (tangential component) onto the plane
///        perpendicular to normal.
/// @param v       Incident vector
/// @param normal  Unit normal vector
/// @return        v - (v.normal)*normal
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
projected(const Float3& v, const Float3& normal) noexcept {
    const float d = dot(v, normal);
    return Float3(v.x - d * normal.x,
                  v.y - d * normal.y,
                  v.z - d * normal.z);
}

/// @brief Generates two tangent vectors from a normal vector.
/// @param normal  Unit normal vector
/// @return        {t1, t2} -- delegates to normal.tangential()
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::tuple<Float3, Float3>
tangential(const Float3& normal) noexcept {
    return normal.tangential();
}

// =============================================================================
// Unary operators
// =============================================================================

/// @brief Unary plus. Returns the vector unchanged.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator+(const Float3& a) noexcept { return a; }

/// @brief Unary minus. Negates the sign of each component.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator-(const Float3& a) noexcept {
    return Float3(-a.x, -a.y, -a.z);
}

// =============================================================================
// Binary arithmetic operators (including mixed scalar/vector forms)
// =============================================================================

/// @brief Scalar + vector (adds a to each component).
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator+(const float a, const Float3& b) noexcept {
    return Float3(a + b.x, a + b.y, a + b.z);
}

/// @brief Vector + scalar.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator+(const Float3& a, const float b) noexcept {
    return Float3(a.x + b, a.y + b, a.z + b);
}

/// @brief Vector + vector (component-wise).
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator+(const Float3& a, const Float3& b) noexcept {
    return Float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/// @brief Vector - scalar.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator-(const Float3& a, const float b) noexcept {
    return Float3(a.x - b, a.y - b, a.z - b);
}

/// @brief Scalar - vector.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator-(const float a, const Float3& b) noexcept {
    return Float3(a - b.x, a - b.y, a - b.z);
}

/// @brief Vector - vector (component-wise).
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator-(const Float3& a, const Float3& b) noexcept {
    return Float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/// @brief Vector * scalar.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator*(const Float3& a, const float b) noexcept {
    return Float3(a.x * b, a.y * b, a.z * b);
}

/// @brief Scalar * vector.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator*(const float a, const Float3& b) noexcept {
    return Float3(a * b.x, a * b.y, a * b.z);
}

/// @brief Vector * vector (component-wise Hadamard product).
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator*(const Float3& a, const Float3& b) noexcept {
    return Float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

/// @brief Vector / scalar. Pre-computes the reciprocal and applies it as 3 multiplications.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator/(const Float3& a, const float b) noexcept {
    const float inv = 1.0f / b; // One division -> three multiplications
    return Float3(a.x * inv, a.y * inv, a.z * inv);
}

/// @brief Scalar / vector (applies the reciprocal component-wise).
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator/(const float a, const Float3& b) noexcept {
    return Float3(a / b.x, a / b.y, a / b.z);
}

/// @brief Vector / vector (component-wise).
/// @warning Produces inf/NaN if a component is 0.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator/(const Float3& a, const Float3& b) noexcept {
    return Float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

// =============================================================================
// Component-wise comparison operators -> return Bool3
// =============================================================================
// Returns a Bool3 instead of a scalar bool, so it can be used for SIMD
// mask selection and similar purposes.

/// @brief Component-wise less-than (`<`) comparison. Returns a Bool3.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator<(const Float3& a, const Float3& b) noexcept {
    return Bool3 { a.x < b.x, a.y < b.y, a.z < b.z };
}

/// @brief Component-wise less-than-or-equal (`<=`) comparison.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator<=(const Float3& a, const Float3& b) noexcept {
    return Bool3 { a.x <= b.x, a.y <= b.y, a.z <= b.z };
}

/// @brief Component-wise greater-than (`>`) comparison.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator>(const Float3& a, const Float3& b) noexcept {
    return Bool3 { a.x > b.x, a.y > b.y, a.z > b.z };
}

/// @brief Component-wise greater-than-or-equal (`>=`) comparison.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator>=(const Float3& a, const Float3& b) noexcept {
    return Bool3 { a.x >= b.x, a.y >= b.y, a.z >= b.z };
}

// =============================================================================
// Math utility free functions
// =============================================================================

/// @brief Returns the component-wise minimum vector.
/// @return {min(a.x,b.x), min(a.y,b.y), min(a.z,b.z)}
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
min(const Float3& a, const Float3& b) noexcept {
    return Float3((a.x < b.x) ? a.x : b.x,
                  (a.y < b.y) ? a.y : b.y,
                  (a.z < b.z) ? a.z : b.z);
}

/// @brief Returns the component-wise maximum vector.
/// @return {max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)}
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
max(const Float3& a, const Float3& b) noexcept {
    return Float3((a.x > b.x) ? a.x : b.x,
                  (a.y > b.y) ? a.y : b.y,
                  (a.z > b.z) ? a.z : b.z);
}

/// @brief Alias for min(). Component-wise min.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
cmin(const Float3& a, const Float3& b) noexcept { return min(a, b); }

/// @brief Alias for max(). Component-wise max.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
cmax(const Float3& a, const Float3& b) noexcept { return max(a, b); }

/// @brief Clamps each component to the range [low, high].
/// @param v     Input vector
/// @param low   Lower bound vector
/// @param high  Upper bound vector
/// @return      Component-wise clamp(v, low, high)
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
clamp(const Float3& v, const Float3& low, const Float3& high) noexcept {
    return min(max(v, low), high); // max first -> guarantees low bound, then min -> guarantees high bound
}

/// @brief Applies ceil (round up) to each component.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
ceil(const Float3& a) noexcept {
    return Float3(std::ceil(a.x), std::ceil(a.y), std::ceil(a.z));
}

/// @brief Applies floor (round down) to each component.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
floor(const Float3& a) noexcept {
    return Float3(std::floor(a.x), std::floor(a.y), std::floor(a.z));
}

/// @brief Returns the absolute value of each component.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
abs(const Float3& v) noexcept {
    return Float3(std::abs(v.x), std::abs(v.y), std::abs(v.z));
}

/// @brief Checks whether all components are finite (excludes inf, NaN).
/// @return true if all three components pass isfinite
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
isfinite(const Float3& v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// =============================================================================
// XY-plane projection utilities
// =============================================================================
// Ignores the z component and computes dot product / length in the xy
// plane. Used for cylindrical coordinate systems or 2D projection
// calculations.

/// @brief Dot product using only the xy components. `a.x*b.x + a.y*b.y` (uses FMA).
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
xy_dot(const Float3& a, const Float3& b) noexcept {
    return std::fma(a.y, b.y, a.x * b.x); // z is ignored
}

/// @brief Squared length in the xy plane. `v.x^2 + v.y^2`.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
xy_length_squared(const Float3& v) noexcept {
    return xy_dot(v, v);
}

/// @brief Length in the xy plane. `sqrt(v.x^2 + v.y^2)`.
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
xy_length(const Float3& v) noexcept {
    return std::sqrt(xy_length_squared(v));
}

// =============================================================================
// Safe normalization utilities
// =============================================================================

/// @brief Normalizes only if the length exceeds min_length_squared.
///
/// Returns a fallback for zero or very short vectors to avoid producing NaN.
///
/// @param v                  Vector to normalize
/// @param fallback           Vector to return if the length is insufficient
/// @param min_length_squared Minimum squared-length threshold for normalizing (default 0)
/// @return                   Unit vector or fallback
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
normalized_or(const Float3& v,
              const Float3& fallback,
              const float min_length_squared = 0.0f) noexcept {
    const float len2 = v.length_squared();
    if (!(len2 > min_length_squared)) {
        return fallback; // Too short: return the fallback
    }
    return v * (1.0f / std::sqrt(len2));
}

/// @brief Normalizes only the xy components. The z component is set to 0.
///
/// @param v                  Input vector (z is ignored)
/// @param fallback           Vector to return if the length is insufficient
/// @param min_length_squared Minimum squared-length threshold for normalizing
/// @return                   xy unit vector (z=0) or fallback
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
xy_normalized_or(const Float3& v,
                 const Float3& fallback,
                 const float min_length_squared = 0.0f) noexcept {
    const float len2 = xy_length_squared(v);
    if (!(len2 > min_length_squared)) {
        return fallback;
    }
    const float inv = 1.0f / std::sqrt(len2);
    return Float3(v.x * inv, v.y * inv, 0.0f); // z is fixed to 0
}

/// @brief Removes (rejects) the component of v along the normal direction.
///
/// Synonymous with `projected`, but provided separately since "reject"
/// is a more common term in the graphics community.
///
/// @param v       Input vector
/// @param normal  Unit normal vector
/// @return        v - (v.normal)*normal
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
reject(const Float3& v, const Float3& normal) noexcept {
    return v - normal * v.dot(normal);
}

// =============================================================================
// Orthonormal basis construction
// =============================================================================

/// @brief Constructs an orthonormal basis (unit_normal, tangent, bitangent)
///        from a normal vector.
///
/// Algorithm:
/// 1. Normalize normal -> unit_normal
/// 2. Choose (0,0,1) as the helper axis if unit_normal.z < 0.9, otherwise
///    choose (0,1,0)
///    (selects an axis that is not nearly parallel to unit_normal)
/// 3. tangent = axis x unit_normal, normalized
/// 4. bitangent = unit_normal x tangent, normalized
///
/// @param[in]  normal               Input normal vector (need not be normalized)
/// @param[out] unit_normal          Normalized normal
/// @param[out] tangent              First tangent vector
/// @param[out] bitangent            Second tangent vector (normal x tangent)
/// @param[in]  min_length_squared   Squared-length threshold for each step (default 0)
/// @return     true on successful basis construction, false on degeneracy (e.g. zero vector)
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
orthonormal_basis(const Float3& normal,
                  Float3& unit_normal,
                  Float3& tangent,
                  Float3& bitangent,
                  const float min_length_squared = 0.0f) noexcept {
    // Step 1: check normal's length and normalize
    const float normal_length_squared = normal.length_squared();
    if (!(normal_length_squared > min_length_squared)) {
        return false; // Degenerate: length is too small
    }
    unit_normal = normal * (1.0f / std::sqrt(normal_length_squared));

    // Step 2: choose a helper axis that is linearly independent of unit_normal
    // If |z| < 0.9, the z-axis is sufficiently different -> use the z-axis
    // Otherwise (unit_normal is close to the z-axis), use the y-axis
    const Float3 axis = std::abs(unit_normal.z) < 0.9f
        ? Float3(0.0f, 0.0f, 1.0f)
        : Float3(0.0f, 1.0f, 0.0f);

    // Step 3: tangent = axis x unit_normal -> perpendicular to normal
    tangent                            = axis.cross(unit_normal);
    const float tangent_length_squared = tangent.length_squared();
    if (!(tangent_length_squared > min_length_squared)) {
        return false; // axis and unit_normal are parallel (degenerate case)
    }
    tangent *= 1.0f / std::sqrt(tangent_length_squared);

    // Step 4: bitangent = unit_normal x tangent -> completes the third axis
    bitangent                            = unit_normal.cross(tangent);
    const float bitangent_length_squared = bitangent.length_squared();
    if (!(bitangent_length_squared > min_length_squared)) {
        return false;
    }
    bitangent *= 1.0f / std::sqrt(bitangent_length_squared);
    return true;
}

/// @brief Overload returning only tangent and bitangent, without the unit_normal output.
///
/// A convenience function for when unit_normal is not needed.
///
/// @param[in]  normal               Input normal vector
/// @param[out] tangent              First tangent vector
/// @param[out] bitangent            Second tangent vector
/// @param[in]  min_length_squared   Squared-length threshold
/// @return     true on successful basis construction
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
orthonormal_basis(const Float3& normal,
                  Float3& tangent,
                  Float3& bitangent,
                  const float min_length_squared = 0.0f) noexcept {
    Float3 unit_normal;
    return orthonormal_basis(normal, unit_normal, tangent, bitangent, min_length_squared);
}

/// @brief Finds a unit vector perpendicular to normal, using seed as a hint.
///
/// Order of attempts:
/// 1. `tangent = normal x seed` -- if seed is not parallel to normal,
///    normalize and return immediately
/// 2. Otherwise, compute a stable tangent via `orthonormal_basis`
/// 3. If both fail, return the (1, 0, 0) fallback
///
/// @param normal               Unit normal vector
/// @param seed                 Hint vector (should not be parallel to normal)
/// @param min_length_squared   Threshold (default 0)
/// @return                     Unit vector perpendicular to normal
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
orthogonal_unit_vector(const Float3& normal,
                       const Float3& seed,
                       const float min_length_squared = 0.0f) noexcept {
    const Float3 fallback(1.0f, 0.0f, 0.0f); // Final fallback: +X axis

    // Attempt 1: if seed is a valid hint, compute quickly via cross product
    Float3 tangent = normal.cross(seed);
    if (tangent.length_squared() > min_length_squared) {
        return normalized_or(tangent, fallback, min_length_squared);
    }

    // Attempt 2: seed is parallel to normal -> compute a stable ONB
    Float3 unit_normal;
    Float3 bitangent;
    if (orthonormal_basis(normal, unit_normal, tangent, bitangent, min_length_squared)) {
        return tangent; // Return the first tangent of the ONB
    }

    return fallback; // Fully degenerate case
}

// =============================================================================
// Spherical-to-Cartesian coordinate conversion
// =============================================================================

/// @brief Generates a spherical direction vector around a unit axis.
///
/// Converts spherical coordinates, with unit_axis as the polar axis
/// (north pole), into Cartesian coordinates:
///
/// ```
/// result = unit_axis * cos_theta
///        + (tangent * cos_phi + bitangent * sin_phi) * sin_theta
/// ```
///
/// The tangent basis is generated automatically via unit_axis.tangential().
///
/// @param unit_axis  Unit polar-axis vector
/// @param cos_theta  Cosine of the polar angle (theta). theta=0 means the direction of unit_axis.
/// @param phi        Azimuthal angle (phi) [radians]
/// @return           Unit vector in the spherical direction
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
spherical_direction(const Float3& unit_axis, const float cos_theta, const float phi) noexcept {
    const auto tangents    = unit_axis.tangential(); // Computes {t1, t2}
    const Float3 tangent   = std::get<0>(tangents);
    const Float3 bitangent = std::get<1>(tangents);

    // sin_theta = sqrt(1 - cos^2(theta)), uses sqrt_nonnegative to guard against negative input
    const float sin_theta = sqrt_nonnegative(1.0f - cos_theta * cos_theta);
    const float cos_phi   = std::cos(phi);
    const float sin_phi   = std::sin(phi);

    // Polar-axis component + equatorial-plane component
    return unit_axis * cos_theta + (tangent * cos_phi + bitangent * sin_phi) * sin_theta;
}

/// @brief Converts standard spherical coordinates (Z-axis based) into Cartesian coordinates.
///
/// Standard spherical coordinate system with +Z as the polar axis:
/// ```
/// x = sin(theta) * cos(phi)
/// y = sin(theta) * sin(phi)
/// z = cos(theta)
/// ```
///
/// @param cos_theta  Cosine of the polar angle theta (range: [-1, 1])
/// @param phi        Azimuthal angle phi [radians]
/// @return           Direction vector on the unit sphere
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
spherical_direction(const float cos_theta, const float phi) noexcept {
    // sin_theta = sqrt(1 - cos^2(theta))
    const float sin_theta = sqrt_nonnegative(1.0f - cos_theta * cos_theta);
    return Float3(sin_theta * std::cos(phi),   // x
                  sin_theta * std::sin(phi),   // y
                  cos_theta);                  // z
}

} // namespace atlas