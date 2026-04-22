#pragma once

namespace atlas::system {

template <typename T>
T
WendlandQuinticSphKernel<T>::density_weight(const T radius,
                                            const T smoothing_length) noexcept {
    // Validate kernel domain.
    //
    // Requirements:
    // - smoothing_length must be strictly positive
    // - radius must be non-negative
    // - radius must lie within compact support [0, h]
    //
    // Outside this region, the Wendland kernel contributes zero.
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }

    // Normalized radial coordinate:
    //     q = r / h
    //
    // This maps the physical distance into the canonical kernel domain [0, 1].
    const T q = radius / smoothing_length;

    // Support factor:
    //     (1 - q)
    //
    // Wendland kernels are constructed so that the function smoothly goes to zero
    // at q = 1, avoiding discontinuities at the support boundary.
    const T one_minus_q = T(1) - q;

    // Normalization coefficient for the 3D Wendland quintic kernel:
    //     α = 21 / (2π h^3)
    //
    // This ensures the kernel integrates to 1 over its support volume.
    const T alpha = static_cast<T>(21.0 / (2.0 * atlas::pi))
        / (smoothing_length * smoothing_length * smoothing_length);

    // Kernel form:
    //     W(q) = α * (1 - q)^4 * (1 + 4q)
    //
    // Properties:
    // - C2 continuous (twice differentiable)
    // - strictly positive inside support
    // - no negative lobes → improved stability vs spline kernels
    return alpha
        * one_minus_q * one_minus_q * one_minus_q * one_minus_q
        * (T(1) + T(4) * q);
}

template <typename T>
Vector3<T>
WendlandQuinticSphKernel<T>::pressure_gradient(const Vector3<T>& delta,
                                               const T radius,
                                               const T smoothing_length) noexcept {
    // Validate kernel domain.
    //
    // Requirements:
    // - smoothing_length must be positive
    // - radius must be strictly positive (division by radius occurs)
    // - radius must lie within compact support
    //
    // The strict radius > 0 check prevents division by zero at particle overlap.
    if (!(smoothing_length > T(0)) || !(radius > T(0)) || radius > smoothing_length) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // Normalized radial coordinate.
    const T q = radius / smoothing_length;

    // Support factor.
    const T one_minus_q = T(1) - q;

    // Gradient normalization coefficient:
    //     α = -210 / (π h^4)
    //
    // The negative sign reflects that the gradient points toward decreasing radius
    // in the scalar radial formulation.
    const T alpha = static_cast<T>(-210.0 / atlas::pi)
        / (smoothing_length * smoothing_length
           * smoothing_length * smoothing_length);

    // Radial derivative of the kernel:
    //
    // Derived from:
    //     W(q) = (1 - q)^4 (1 + 4q)
    //
    // After differentiation, the radial dependence yields:
    //     dW/dr ∝ q (1 - q)^3
    //
    // This expression is stable and avoids singular behavior near q = 0.
    const T radial_derivative =
        alpha * q * one_minus_q * one_minus_q * one_minus_q;

    // Convert scalar radial derivative into vector gradient:
    //
    //     ∇W = (dW/dr) * (delta / r)
    //
    // where delta is the displacement vector between particles.
    return delta * (radial_derivative / radius);
}

template <typename T>
T
WendlandQuinticSphKernel<T>::viscosity_laplacian(const T radius,
                                                 const T smoothing_length) noexcept {
    // Validate kernel domain.
    //
    // Requirements:
    // - smoothing_length must be positive
    // - radius must be non-negative
    // - radius must lie within compact support
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }

    // Normalized radial coordinate.
    const T q = radius / smoothing_length;

    // Support factor.
    const T one_minus_q = T(1) - q;

    // Laplacian normalization coefficient:
    //     α = 210 / (π h^5)
    //
    // Higher power of h reflects the second spatial derivative scaling.
    const T alpha = static_cast<T>(210.0 / atlas::pi)
        / (smoothing_length * smoothing_length * smoothing_length
           * smoothing_length * smoothing_length);

    // Laplacian form:
    //
    // Derived from second derivatives of the Wendland quintic kernel.
    // A simplified stable form:
    //     ∇²W ∝ (1 - q)^2 (1 - 4q)
    //
    // Properties:
    // - smooth near q = 0
    // - avoids excessive noise compared to spline Laplacians
    return alpha
        * one_minus_q * one_minus_q
        * (T(1) - T(4) * q);
}

} // namespace atlas::system