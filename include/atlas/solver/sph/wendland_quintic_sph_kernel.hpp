#pragma once

namespace atlas::system {

template <typename T>
T
WendlandQuinticSphKernel<T>::density_weight(const T radius,
                                            const T smoothing_length) noexcept {
    // The kernel is compactly supported for 0 <= r <= h.
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }

    // Normalized radius: q = r / h.
    const T q           = radius / smoothing_length;
    const T one_minus_q = T(1) - q;

    // 3D Wendland quintic normalization factor.
    const T alpha = static_cast<T>(21.0 / (2.0 * atlas::pi))
        / (smoothing_length * smoothing_length * smoothing_length);

    // W(r, h) = alpha * (1 - q)^4 * (1 + 4q).
    return alpha
        * one_minus_q * one_minus_q * one_minus_q * one_minus_q
        * (T(1) + T(4) * q);
}

template <typename T>
Vector3<T>
WendlandQuinticSphKernel<T>::pressure_gradient(const Vector3<T>& delta,
                                               const T radius,
                                               const T smoothing_length) noexcept {
    // The gradient is undefined at r = 0 and zero outside the support radius.
    if (!(smoothing_length > T(0)) || !(radius > T(0)) || radius > smoothing_length) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // Normalized radius: q = r / h.
    const T q           = radius / smoothing_length;
    const T one_minus_q = T(1) - q;

    // Radial derivative coefficient for the 3D Wendland quintic kernel.
    const T alpha = static_cast<T>(-210.0 / atlas::pi)
        / (smoothing_length * smoothing_length
           * smoothing_length * smoothing_length);

    // dW/dr = alpha * q * (1 - q)^3.
    const T radial_derivative = alpha * q * one_minus_q * one_minus_q * one_minus_q;

    // Convert the radial derivative into a vector gradient: grad W = (dW/dr) * delta / r.
    return delta * (radial_derivative / radius);
}

template <typename T>
T
WendlandQuinticSphKernel<T>::viscosity_laplacian(const T radius,
                                                 const T smoothing_length) noexcept {
    // The kernel Laplacian is compactly supported for 0 <= r <= h.
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }

    // Normalized radius: q = r / h.
    const T q           = radius / smoothing_length;
    const T one_minus_q = T(1) - q;

    // 3D Wendland quintic Laplacian coefficient.
    const T alpha = static_cast<T>(210.0 / atlas::pi)
        / (smoothing_length * smoothing_length * smoothing_length
           * smoothing_length * smoothing_length);

    // Laplacian approximation used for viscosity: alpha * (1 - q)^2 * (1 - 4q).
    return alpha
        * one_minus_q * one_minus_q
        * (T(1) - T(4) * q);
}

} // namespace atlas::systemㅂ