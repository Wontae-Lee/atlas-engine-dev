#pragma once
namespace atlas::system {
template <typename T>
T
WendlandQuinticSphKernel<T>::density_weight(const T radius,
                                            const T smoothing_length) noexcept {
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }
    const T q           = radius / smoothing_length;
    const T one_minus_q = T(1) - q;
    const T alpha       = static_cast<T>(21.0 / (2.0 * atlas::pi))
        / (smoothing_length * smoothing_length * smoothing_length);
    return alpha
        * one_minus_q * one_minus_q * one_minus_q * one_minus_q
        * (T(1) + T(4) * q);
}

template <typename T>
Vector3<T>
WendlandQuinticSphKernel<T>::pressure_gradient(const Vector3<T>& delta,
                                               const T radius,
                                               const T smoothing_length) noexcept {
    if (!(smoothing_length > T(0)) || !(radius > T(0)) || radius > smoothing_length) {
        return Vector3<T>(T(0), T(0), T(0));
    }
    const T q           = radius / smoothing_length;
    const T one_minus_q = T(1) - q;
    const T alpha       = static_cast<T>(-210.0 / atlas::pi)
        / (smoothing_length * smoothing_length
           * smoothing_length * smoothing_length);
    const T radial_derivative = alpha * q * one_minus_q * one_minus_q * one_minus_q;
    return delta * (radial_derivative / radius);
}

template <typename T>
T
WendlandQuinticSphKernel<T>::viscosity_laplacian(const T radius,
                                                 const T smoothing_length) noexcept {
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }
    const T q           = radius / smoothing_length;
    const T one_minus_q = T(1) - q;
    const T alpha       = static_cast<T>(210.0 / atlas::pi)
        / (smoothing_length * smoothing_length * smoothing_length
           * smoothing_length * smoothing_length);
    return alpha
        * one_minus_q * one_minus_q
        * (T(1) - T(4) * q);
}

}