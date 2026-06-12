#pragma once

namespace atlas {

template <typename T>
T
StandardSphKernel<T>::density_weight(const T radius, const T cell_size) noexcept {
    // Reject invalid support radius, invalid cell size, and samples outside the kernel support.
    if (!(cell_size > T(0)) || !(radius >= T(0)) || radius > cell_size) {
        return T(0);
    }

    // Precompute h^2 for the poly6 kernel support term.
    const T cell_size_squared = cell_size * cell_size;

    // Poly6 support term h^2 - r^2.
    const T support = cell_size_squared - radius * radius;

    // 3D poly6 kernel normalization coefficient.
    const T coeff = static_cast<T>(315.0 / (64.0 * atlas::pi));

    // Return W_poly6(r, h) = 315 / (64 pi h^9) * (h^2 - r^2)^3.
    return coeff * support * support * support
        / (cell_size * cell_size * cell_size
           * cell_size * cell_size * cell_size
           * cell_size * cell_size * cell_size);
}

template <typename T>
Vector3<T>
StandardSphKernel<T>::pressure_gradient(const Vector3<T>& delta,
                                        const T radius,
                                        const T cell_size) noexcept {
    // Reject invalid cell size, zero-length direction, and samples outside the kernel support.
    if (!(cell_size > T(0)) || !(radius > T(0)) || radius > cell_size) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // 3D spiky kernel gradient coefficient.
    const T coeff = static_cast<T>(-45.0 / atlas::pi);

    // Squared compact-support distance for the spiky pressure gradient.
    const T support = (cell_size - radius) * (cell_size - radius);

    // Convert the scalar radial derivative into a vector gradient along delta.
    return delta * (coeff * support / (cell_size * cell_size * cell_size * cell_size * cell_size * cell_size * radius));
}

template <typename T>
T
StandardSphKernel<T>::viscosity_laplacian(const T radius, const T cell_size) noexcept {
    // Reject invalid support radius, invalid cell size, and samples outside the kernel support.
    if (!(cell_size > T(0)) || !(radius >= T(0)) || radius > cell_size) {
        return T(0);
    }

    // 3D viscosity kernel Laplacian coefficient.
    const T coeff = static_cast<T>(45.0 / atlas::pi);

    // Return the viscosity Laplacian, which decays linearly to zero at r = h.
    return coeff * (cell_size - radius)
        / (cell_size * cell_size * cell_size
           * cell_size * cell_size * cell_size);
}

} // namespace atlas