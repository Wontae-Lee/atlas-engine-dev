#pragma once
namespace atlas::system {
template <typename T>
T
StandardSphKernel<T>::density_weight(const T radius, const T smoothing_length) noexcept {
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }
    const T smoothing_length_squared = smoothing_length * smoothing_length;
    const T support                  = smoothing_length_squared - radius * radius;
    const T coeff                    = static_cast<T>(315.0 / (64.0 * atlas::pi));
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
    if (!(smoothing_length > T(0)) || !(radius > T(0)) || radius > smoothing_length) {
        return Vector3<T>(T(0), T(0), T(0));
    }
    const T coeff   = static_cast<T>(-45.0 / atlas::pi);
    const T support = (smoothing_length - radius) * (smoothing_length - radius);
    return delta * (coeff * support / (smoothing_length * smoothing_length * smoothing_length * smoothing_length * smoothing_length * smoothing_length * radius));
}

template <typename T>
T
StandardSphKernel<T>::viscosity_laplacian(const T radius, const T smoothing_length) noexcept {
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }
    const T coeff = static_cast<T>(45.0 / atlas::pi);
    return coeff * (smoothing_length - radius)
        / (smoothing_length * smoothing_length * smoothing_length
           * smoothing_length * smoothing_length * smoothing_length);
}

}