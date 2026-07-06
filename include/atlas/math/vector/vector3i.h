#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/vector/bool3.h>
#include <atlas/math/vector/vector3.h>

#include <cstddef>

namespace atlas {

class Vector3i {
public:
    int x;

    int y;

    int z;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Vector3i() noexcept
        : x(0)
        , y(0)
        , z(0) { }

    constexpr Vector3i(const Vector3i&) noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Vector3i(const int s) noexcept
        : x(s)
        , y(s)
        , z(s) { }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Vector3i(const int x_, const int y_, const int z_) noexcept
        : x(x_)
        , y(y_)
        , z(z_) { }

    ~Vector3i() noexcept = default;

    Vector3i&
    operator=(const Vector3i&) noexcept = default;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size() noexcept {
        return 3;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const int*
    data() const noexcept {
        return &x;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int*
    data() noexcept {
        return &x;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const int&
    operator[](const std::size_t i) const noexcept {
        return (&x)[i];
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int&
    operator[](const std::size_t i) noexcept {
        return (&x)[i];
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_zero() noexcept {
        x = y = z = 0;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3i&
    operator+=(const Vector3i& v) noexcept {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3i&
    operator-=(const Vector3i& v) noexcept {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3i&
    operator*=(const int s) noexcept {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Vector3i& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Vector3i& other) const noexcept {
        return !(*this == other);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
    min() const noexcept {
        return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
    max() const noexcept {
        return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
    }
};


ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3i
operator+(const Vector3i& a, const Vector3i& b) noexcept {
    return Vector3i(a.x + b.x, a.y + b.y, a.z + b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3i
operator-(const Vector3i& a, const Vector3i& b) noexcept {
    return Vector3i(a.x - b.x, a.y - b.y, a.z - b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3i
operator-(const Vector3i& a) noexcept {
    return Vector3i(-a.x, -a.y, -a.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3i
operator*(const Vector3i& a, const int s) noexcept {
    return Vector3i(a.x * s, a.y * s, a.z * s);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3i
operator*(const int s, const Vector3i& a) noexcept {
    return a * s;
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator<(const Vector3i& a, const Vector3i& b) noexcept {
    return Bool3 { a.x < b.x, a.y < b.y, a.z < b.z };
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator<=(const Vector3i& a, const Vector3i& b) noexcept {
    return Bool3 { a.x <= b.x, a.y <= b.y, a.z <= b.z };
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator>(const Vector3i& a, const Vector3i& b) noexcept {
    return Bool3 { a.x > b.x, a.y > b.y, a.z > b.z };
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator>=(const Vector3i& a, const Vector3i& b) noexcept {
    return Bool3 { a.x >= b.x, a.y >= b.y, a.z >= b.z };
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3i
min(const Vector3i& a, const Vector3i& b) noexcept {
    return Vector3i((a.x < b.x) ? a.x : b.x,
                    (a.y < b.y) ? a.y : b.y,
                    (a.z < b.z) ? a.z : b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3i
max(const Vector3i& a, const Vector3i& b) noexcept {
    return Vector3i((a.x > b.x) ? a.x : b.x,
                    (a.y > b.y) ? a.y : b.y,
                    (a.z > b.z) ? a.z : b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3i
clamp(const Vector3i& v, const Vector3i& low, const Vector3i& high) noexcept {
    return min(max(v, low), high);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3i
to_vector3i(const Vector3& v) noexcept {
    return Vector3i(static_cast<int>(v.x),
                    static_cast<int>(v.y),
                    static_cast<int>(v.z));
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
to_vector3(const Vector3i& v) noexcept {
    return Vector3(static_cast<float>(v.x),
                   static_cast<float>(v.y),
                   static_cast<float>(v.z));
}

}
