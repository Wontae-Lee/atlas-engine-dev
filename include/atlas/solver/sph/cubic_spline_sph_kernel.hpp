#pragma once

namespace atlas {

template <typename T>
T
CubicSplineSphKernel<T>::density_weight(const T radius, const T cell_size) noexcept {

    if (!(cell_size > T(0)) || !(radius >= T(0)) || radius > cell_size) {
        return T(0);
    }

    const T q = radius / cell_size;

    const T alpha = static_cast<T>(1.0 / atlas::pi)
        / (cell_size * cell_size * cell_size);

    if (q < T(0.5)) {

        return alpha * (T(6) * q * q * q - T(6) * q * q + T(1));
    }

    const T one_minus_q = T(1) - q;
    return alpha * (T(2) * one_minus_q * one_minus_q * one_minus_q);
}

template <typename T>
Vector3<T>
CubicSplineSphKernel<T>::pressure_gradient(const Vector3<T>& delta,
                                           const T radius,
                                           const T cell_size) noexcept {

    if (!(cell_size > T(0)) || !(radius > T(0)) || radius > cell_size) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    const T q = radius / cell_size;

    const T alpha = static_cast<T>(6.0 / atlas::pi)
        / (cell_size * cell_size * cell_size * cell_size);

    T radial_derivative = T(0);

    if (q < T(0.5)) {

        radial_derivative = alpha * (T(3) * q * q - T(2) * q);
    } else {

        const T one_minus_q = T(1) - q;
        radial_derivative   = -alpha * one_minus_q * one_minus_q;
    }

    return delta * (radial_derivative / radius);
}

template <typename T>
T
CubicSplineSphKernel<T>::viscosity_laplacian(const T radius, const T cell_size) noexcept {

    if (!(cell_size > T(0)) || !(radius >= T(0)) || radius > cell_size) {
        return T(0);
    }

    const T q = radius / cell_size;

    const T alpha = static_cast<T>(6.0 / atlas::pi)
        / (cell_size * cell_size * cell_size * cell_size * cell_size);

    if (q < T(0.5)) {

        return alpha * (T(6) * q - T(2));
    }

    return alpha * (T(2) - T(2) * q);
}

}