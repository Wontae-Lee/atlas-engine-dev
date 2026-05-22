#pragma once

namespace atlas::system {

template <typename T>
T
StandardSphKernel<T>::density_weight(const T radius, const T smoothing_length) noexcept {
    // Reject invalid support radius, invalid smoothing length, and samples outside the kernel support.
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }

    // Precompute h^2 for the poly6 kernel support term.
    const T smoothing_length_squared = smoothing_length * smoothing_length;

    // Poly6 support term h^2 - r^2.
    const T support = smoothing_length_squared - radius * radius;

    // 3D poly6 kernel normalization coefficient.
    const T coeff = static_cast<T>(315.0 / (64.0 * atlas::pi));

    // Return W_poly6(r, h) = 315 / (64 pi h^9) * (h^2 - r^2)^3.
    return coeff * support * support * support
        / (smoothing_length * smoothing_length * smoothing_length
           * smoothing_length * smoothing_length * smoothing_length
           * smoothing_length * smoothing_length * smoothing_length);
}

template <typename T>
Vector3<T>
StandardSphKernel<T>::pressure_gradient(const Vector3<T>& delta,
                                        const T radius,
                                        const T smoothing_length) noexcept {
    // Reject invalid smoothing length, zero-length direction, and samples outside the kernel support.
    if (!(smoothing_length > T(0)) || !(radius > T(0)) || radius > smoothing_length) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // 3D spiky kernel gradient coefficient.
    const T coeff = static_cast<T>(-45.0 / atlas::pi);

    // Squared compact-support distance for the spiky pressure gradient.
    const T support = (smoothing_length - radius) * (smoothing_length - radius);

    // Convert the scalar radial derivative into a vector gradient along delta.
    return delta * (coeff * support / (smoothing_length * smoothing_length * smoothing_length * smoothing_length * smoothing_length * smoothing_length * radius));
}

template <typename T>
T
StandardSphKernel<T>::viscosity_laplacian(const T radius, const T smoothing_length) noexcept {
    // Reject invalid support radius, invalid smoothing length, and samples outside the kernel support.
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }

    // 3D viscosity kernel Laplacian coefficient.
    const T coeff = static_cast<T>(45.0 / atlas::pi);

    // Return the viscosity Laplacian, which decays linearly to zero at r = h.
    return coeff * (smoothing_length - radius)
        / (smoothing_length * smoothing_length * smoothing_length
           * smoothing_length * smoothing_length * smoothing_length);
}

} // namespace atlas::system