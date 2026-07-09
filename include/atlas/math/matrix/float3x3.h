#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/vector/float3.h>

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <utility>

namespace atlas {

class Float3x3 {
public:
    union {
        struct {
            float m00, m01, m02;
            float m10, m11, m12;
            float m20, m21, m22;
        };
        float _data[9];
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Float3x3() noexcept
        : m00(0.0f)
        , m01(0.0f)
        , m02(0.0f)
        , m10(0.0f)
        , m11(0.0f)
        , m12(0.0f)
        , m20(0.0f)
        , m21(0.0f)
        , m22(0.0f) { }

    constexpr Float3x3(const Float3x3&) noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Float3x3(const float s) noexcept
        : m00(s)
        , m01(0.0f)
        , m02(0.0f)
        , m10(0.0f)
        , m11(s)
        , m12(0.0f)
        , m20(0.0f)
        , m21(0.0f)
        , m22(s) { }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Float3x3(const float a00, const float a01, const float a02,
                                                           const float a10, const float a11, const float a12,
                                                           const float a20, const float a21, const float a22) noexcept
        : m00(a00)
        , m01(a01)
        , m02(a02)
        , m10(a10)
        , m11(a11)
        , m12(a12)
        , m20(a20)
        , m21(a21)
        , m22(a22) { }

    ATLAS_HOST ATLAS_FORCE_INLINE explicit Float3x3(std::initializer_list<float> list) noexcept {
        const float* it = list.begin();
        for (int i = 0; i < 9; ++i) {
            _data[i] = (it != list.end()) ? *it++ : 0.0f;
        }
    }

    ~Float3x3() noexcept = default;

    Float3x3&
    operator=(const Float3x3&) noexcept = default;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    rows() noexcept {
        return 3;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    cols() noexcept {
        return 3;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size() noexcept {
        return 9;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float*
    data() const noexcept {
        return _data;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float*
    data() noexcept {
        return _data;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    operator[](const std::size_t i) const noexcept {
        return _data[i];
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    operator[](const std::size_t i) noexcept {
        return _data[i];
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    at(const std::size_t r, const std::size_t c) const noexcept {
        return _data[r * 3 + c];
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    at(const std::size_t r, const std::size_t c) noexcept {
        return _data[r * 3 + c];
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    operator()(const std::size_t r, const std::size_t c) const noexcept {
        return _data[r * 3 + c];
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    operator()(const std::size_t r, const std::size_t c) noexcept {
        return _data[r * 3 + c];
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_zero() noexcept {
        for (int i = 0; i < 9; ++i) _data[i] = 0.0f;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_identity() noexcept {
        set(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set(const float a00, const float a01, const float a02,
        const float a10, const float a11, const float a12,
        const float a20, const float a21, const float a22) noexcept {
        m00 = a00;
        m01 = a01;
        m02 = a02;
        m10 = a10;
        m11 = a11;
        m12 = a12;
        m20 = a20;
        m21 = a21;
        m22 = a22;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const float s) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] += s;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const float s) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] -= s;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    mul(const float s) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] *= s;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    div(const float s) noexcept {
        const float inv = 1.0f / s;
        for (int i = 0; i < 9; ++i) _data[i] *= inv;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const Float3x3& m) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] += m._data[i];
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const Float3x3& m) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] -= m._data[i];
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator+=(const float s) noexcept {
        add(s);
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator-=(const float s) noexcept {
        sub(s);
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator*=(const float s) noexcept {
        mul(s);
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator/=(const float s) noexcept {
        div(s);
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator+=(const Float3x3& m) noexcept {
        add(m);
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator-=(const Float3x3& m) noexcept {
        sub(m);
        return *this;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Float3x3& o) const noexcept {
        for (int i = 0; i < 9; ++i) {
            if (_data[i] != o._data[i]) return false;
        }
        return true;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Float3x3& o) const noexcept {
        return !(*this == o);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    determinant() const noexcept {
        const float ei_fh = m11 * m22 - m12 * m21;
        const float di_fg = m10 * m22 - m12 * m20;
        const float dh_eg = m10 * m21 - m11 * m20;
        return std::fma(m00, ei_fh, std::fma(-m01, di_fg, m02 * dh_eg));
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    trace() const noexcept {
        return m00 + m11 + m22;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    transpose() noexcept {
        std::swap(m01, m10);
        std::swap(m02, m20);
        std::swap(m12, m21);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
    transposed() const noexcept {
        return Float3x3(m00, m10, m20, m01, m11, m21, m02, m12, m22);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    inverse() noexcept {
        Float3x3 out;
        cofactor_inverse(out, determinant());
        *this = out;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
    inversed() const noexcept {
        Float3x3 out = *this;
        out.inverse();
        return out;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    try_inverse(Float3x3& out, const float eps = std::numeric_limits<float>::epsilon()) const noexcept {
        const float det = determinant();
        if (std::abs(det) <= eps) return false;
        cofactor_inverse(out, det);
        return true;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_invertible(const float eps = std::numeric_limits<float>::epsilon()) const noexcept {
        return std::abs(determinant()) > eps;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
    mul(const Float3x3& r) const noexcept {
        const Float3x3& a = *this;
        Float3x3 out;
        out.m00 = std::fma(a.m01, r.m10, std::fma(a.m02, r.m20, a.m00 * r.m00));
        out.m01 = std::fma(a.m01, r.m11, std::fma(a.m02, r.m21, a.m00 * r.m01));
        out.m02 = std::fma(a.m01, r.m12, std::fma(a.m02, r.m22, a.m00 * r.m02));
        out.m10 = std::fma(a.m11, r.m10, std::fma(a.m12, r.m20, a.m10 * r.m00));
        out.m11 = std::fma(a.m11, r.m11, std::fma(a.m12, r.m21, a.m10 * r.m01));
        out.m12 = std::fma(a.m11, r.m12, std::fma(a.m12, r.m22, a.m10 * r.m02));
        out.m20 = std::fma(a.m21, r.m10, std::fma(a.m22, r.m20, a.m20 * r.m00));
        out.m21 = std::fma(a.m21, r.m11, std::fma(a.m22, r.m21, a.m20 * r.m01));
        out.m22 = std::fma(a.m21, r.m12, std::fma(a.m22, r.m22, a.m20 * r.m02));
        return out;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    mul(const Float3& v) const noexcept {
        return Float3(std::fma(m01, v.y, std::fma(m02, v.z, m00 * v.x)),
                      std::fma(m11, v.y, std::fma(m12, v.z, m10 * v.x)),
                      std::fma(m21, v.y, std::fma(m22, v.z, m20 * v.x)));
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    solved(const Float3& b) const noexcept {
        return inversed().mul(b);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    solve(const Float3& b, Float3& x,
          const float eps = std::numeric_limits<float>::epsilon()) const noexcept {
        Float3x3 inv;
        if (!try_inverse(inv, eps)) return false;
        x = inv.mul(b);
        return true;
    }

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    cofactor_inverse(Float3x3& out, const float det) const noexcept {
        const float c00 = (m11 * m22 - m12 * m21);
        const float c01 = -(m10 * m22 - m12 * m20);
        const float c02 = (m10 * m21 - m11 * m20);
        const float c10 = -(m01 * m22 - m02 * m21);
        const float c11 = (m00 * m22 - m02 * m20);
        const float c12 = -(m00 * m21 - m01 * m20);
        const float c20 = (m01 * m12 - m02 * m11);
        const float c21 = -(m00 * m12 - m02 * m10);
        const float c22 = (m00 * m11 - m01 * m10);

        const float inv_det = 1.0f / det;
        out.set(c00 * inv_det, c10 * inv_det, c20 * inv_det, c01 * inv_det, c11 * inv_det, c21 * inv_det, c02 * inv_det, c12 * inv_det, c22 * inv_det);
    }
};

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
identity3x3() noexcept {
    return Float3x3(1.0f);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
zero3x3() noexcept {
    return Float3x3();
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
transpose(const Float3x3& m) noexcept {
    return m.transposed();
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
determinant(const Float3x3& m) noexcept {
    return m.determinant();
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
inverse(const Float3x3& m) noexcept {
    return m.inversed();
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator+(const Float3x3& a, const Float3x3& b) noexcept {
    Float3x3 out;
    for (int i = 0; i < 9; ++i) out._data[i] = a._data[i] + b._data[i];
    return out;
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator-(const Float3x3& a, const Float3x3& b) noexcept {
    Float3x3 out;
    for (int i = 0; i < 9; ++i) out._data[i] = a._data[i] - b._data[i];
    return out;
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator*(const Float3x3& a, const float s) noexcept {
    Float3x3 out;
    for (int i = 0; i < 9; ++i) out._data[i] = a._data[i] * s;
    return out;
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator*(const float s, const Float3x3& a) noexcept {
    return a * s;
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator/(const Float3x3& a, const float s) noexcept {
    return a * (1.0f / s);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator*(const Float3x3& a, const Float3x3& b) noexcept {
    return a.mul(b);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator*(const Float3x3& a, const Float3& v) noexcept {
    return a.mul(v);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
rotate(const Float3x3& matrix, const Float3& input, Float3& output) noexcept {
    const float x = input.x;
    const float y = input.y;
    const float z = input.z;

    output.x = std::fma(matrix.m01, y, std::fma(matrix.m02, z, matrix.m00 * x));
    output.y = std::fma(matrix.m11, y, std::fma(matrix.m12, z, matrix.m10 * x));
    output.z = std::fma(matrix.m21, y, std::fma(matrix.m22, z, matrix.m20 * x));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
rotate_translate(const Float3x3& matrix,
                 const Float3& input,
                 const Float3& offset,
                 Float3& output) noexcept {
    rotate(matrix, input, output);
    output += offset;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
rotate_subtract(const Float3x3& matrix,
                const Float3& input,
                const Float3& offset,
                Float3& output) noexcept {
    rotate(matrix, input - offset, output);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
solve(const Float3x3& A, const Float3& b, Float3& x,
      const float eps = std::numeric_limits<float>::epsilon()) noexcept {
    return A.solve(b, x, eps);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
solve(const Float3x3& A, const Float3& b) noexcept {
    return A.solved(b);
}

}