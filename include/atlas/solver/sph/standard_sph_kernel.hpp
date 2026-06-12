#pragma once

namespace atlas {

template <typename T>
T
StandardSphKernel<T>::density_weight(const T radius, const T cell_size) noexcept {

    if (!(cell_size > T(0)) || !(radius >= T(0)) || radius > cell_size) {
        return T(0);
    }

    const T cell_size_squared = cell_size * cell_size;

    const T support = cell_size_squared - radius * radius;

    const T coeff = static_cast<T>(315.0 / (64.0 * atlas::pi));

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

    if (!(cell_size > T(0)) || !(radius > T(0)) || radius > cell_size) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    const T coeff = static_cast<T>(-45.0 / atlas::pi);

    const T support = (cell_size - radius) * (cell_size - radius);

    return delta * (coeff * support / (cell_size * cell_size * cell_size * cell_size * cell_size * cell_size * radius));
}

template <typename T>
T
StandardSphKernel<T>::viscosity_laplacian(const T radius, const T cell_size) noexcept {

    if (!(cell_size > T(0)) || !(radius >= T(0)) || radius > cell_size) {
        return T(0);
    }

    const T coeff = static_cast<T>(45.0 / atlas::pi);

    return coeff * (cell_size - radius)
        / (cell_size * cell_size * cell_size
           * cell_size * cell_size * cell_size);
}

}