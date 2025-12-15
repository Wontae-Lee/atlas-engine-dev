#pragma once
#include <atlas/math/math.h>
#include <cmath>

namespace atlas::random {
template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE

void
build_orthonormal_basis(const Vector3<T>& n,
                        Vector3<T>& t,
                        Vector3<T>& b) {
    if (std::abs(n.x) > std::abs(n.z)) {
        t = Vector3<T>(-n.y, n.x, T(0));
    } else {
        t = Vector3<T>(T(0), -n.z, n.y);
    }
    t = math::normalize(t);
    b = math::cross(n, t);
}

template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE Vector3<T>

sample_uniform_hemisphere(const Vector3<T>& n, T u1, T u2) {
    const T two_pi      = T(2) * M_PI;
    const T phi         = two_pi * u2;
    const T cos_theta   = T(1) - u1;
    const T sin_theta_2 = T(1) - cos_theta * cos_theta;
    const T sin_theta   = (sin_theta_2 > T(0)) ? std::sqrt(sin_theta_2) : T(0);
    const T cos_phi     = std::cos(phi);
    const T sin_phi     = std::sin(phi);
    const T x           = sin_theta * cos_phi;
    const T y           = sin_theta * sin_phi;
    const T z           = cos_theta;
    Vector3<T> t, b;
    build_orthonormal_basis(n, t, b);
    return x * t + y * b + z * n;
}

template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE Vector3<T>

sample_cosine_hemisphere(const Vector3<T>& n, T u1, T u2) {
    const T two_pi    = T(2) * M_PI;
    const T phi       = two_pi * u1;
    const T cos_theta = std::sqrt(T(1) - u2);
    const T sin_theta = std::sqrt(u2);
    const T cos_phi   = std::cos(phi);
    const T sin_phi   = std::sin(phi);
    const T x         = sin_theta * cos_phi;
    const T y         = sin_theta * sin_phi;
    const T z         = cos_theta;
    Vector3<T> t, b;
    build_orthonormal_basis(n, t, b);
    return x * t + y * b + z * n;
}
}