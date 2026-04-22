#pragma once

namespace atlas::system {

template <typename T>
T
CubicSplineSphKernel<T>::density_weight(const T radius, const T smoothing_length) noexcept {
    // Return zero when the kernel support is not physically or numerically valid.
    //
    // Conditions handled here:
    //   - the smoothing length must be strictly positive,
    //   - the query radius must not be negative,
    //   - the cubic spline kernel has compact support and is zero outside h.
    //
    // Therefore, any sample outside the interval [0, h] contributes no density weight.
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }

    // Normalize the distance by the smoothing length.
    //
    // The cubic spline SPH kernel is typically expressed in terms of the
    // dimensionless radius:
    //
    //     q = r / h
    //
    // where:
    //   - r is the particle separation distance,
    //   - h is the smoothing length.
    const T q = radius / smoothing_length;

    // Compute the three-dimensional normalization factor of the cubic spline kernel.
    //
    // This factor ensures that the kernel integrates to the expected value in 3D
    // and scales correctly with the smoothing length.
    const T alpha = static_cast<T>(1.0 / atlas::pi)
        / (smoothing_length * smoothing_length * smoothing_length);

    // Evaluate the inner branch of the cubic spline kernel for 0 <= q < 1/2.
    //
    // In this region, the density kernel follows the polynomial:
    //
    //     W(q) = alpha * (6q^3 - 6q^2 + 1)
    //
    // This branch produces the highest weights near the center and smoothly
    // decreases as the distance grows.
    if (q < T(0.5)) {
        return alpha * (T(6) * q * q * q - T(6) * q * q + T(1));
    }

    // Evaluate the outer branch of the cubic spline kernel for 1/2 <= q <= 1.
    //
    // In this region, the kernel decays toward zero at the support boundary q = 1:
    //
    //     W(q) = alpha * 2(1 - q)^3
    //
    // This preserves compact support and smooth continuity across the piecewise
    // transition.
    const T one_minus_q = T(1) - q;
    return alpha * (T(2) * one_minus_q * one_minus_q * one_minus_q);
}

template <typename T>
Vector3<T>
CubicSplineSphKernel<T>::pressure_gradient(const Vector3<T>& delta,
                                           const T radius,
                                           const T smoothing_length) noexcept {
    // Return the zero vector when the gradient is not well-defined or outside support.
    //
    // Conditions handled here:
    //   - the smoothing length must be strictly positive,
    //   - the radius must be strictly positive because the final expression
    //     divides by radius when converting a radial derivative into a vector,
    //   - the cubic spline kernel gradient is zero outside the support radius h.
    if (!(smoothing_length > T(0)) || !(radius > T(0)) || radius > smoothing_length) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // Normalize the separation distance by the smoothing length.
    //
    // The piecewise cubic spline derivative is expressed in terms of q = r / h.
    const T q = radius / smoothing_length;

    // Compute the three-dimensional normalization factor for the kernel gradient.
    //
    // Compared with the density kernel normalization, the gradient introduces
    // one additional factor of 1 / h because differentiation is taken with
    // respect to spatial distance.
    const T alpha = static_cast<T>(6.0 / atlas::pi)
        / (smoothing_length * smoothing_length * smoothing_length * smoothing_length);

    // Store the radial derivative dW/dr before converting it to a vector form.
    //
    // The final gradient is obtained by multiplying the radial derivative by
    // the unit direction vector delta / radius.
    T radial_derivative = T(0);

    // Evaluate the derivative of the inner cubic-spline branch for 0 <= q < 1/2.
    //
    // This corresponds to the derivative of:
    //
    //     W(q) = alpha * (6q^3 - 6q^2 + 1)
    //
    // expressed in the scaled form used by this implementation.
    if (q < T(0.5)) {
        radial_derivative = alpha * (T(3) * q * q - T(2) * q);
    } else {
        // Evaluate the derivative of the outer branch for 1/2 <= q <= 1.
        //
        // This branch decays smoothly toward zero at the kernel support boundary.
        const T one_minus_q = T(1) - q;
        radial_derivative   = -alpha * one_minus_q * one_minus_q;
    }

    // Convert the scalar radial derivative into the full vector gradient.
    //
    // Since:
    //
    //     grad W = (dW/dr) * (delta / r)
    //
    // this scales the separation vector by the radial derivative divided by the
    // separation distance.
    return delta * (radial_derivative / radius);
}

template <typename T>
T
CubicSplineSphKernel<T>::viscosity_laplacian(const T radius, const T smoothing_length) noexcept {
    // Return zero when the query is outside the valid kernel support or when
    // the smoothing length is invalid.
    //
    // The viscosity Laplacian is only defined for samples inside the compact
    // support radius and requires a strictly positive smoothing length.
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }

    // Normalize the separation distance by the smoothing length.
    const T q = radius / smoothing_length;

    // Compute the normalization factor used by the Laplacian form of the cubic spline.
    //
    // The Laplacian introduces two additional factors of 1 / h relative to the
    // density kernel because it corresponds to a second spatial derivative.
    const T alpha = static_cast<T>(6.0 / atlas::pi)
        / (smoothing_length * smoothing_length * smoothing_length * smoothing_length * smoothing_length);

    // Evaluate the inner branch of the viscosity Laplacian for 0 <= q < 1/2.
    //
    // This branch captures the curvature of the cubic spline near the kernel center.
    if (q < T(0.5)) {
        return alpha * (T(6) * q - T(2));
    }

    // Evaluate the outer branch of the viscosity Laplacian for 1/2 <= q <= 1.
    //
    // This branch smoothly approaches zero as q approaches the compact-support boundary.
    return alpha * (T(2) - T(2) * q);
}

} // namespace atlas::system