#pragma once

namespace atlas::system {

template <typename T>
T
StandardSphKernel<T>::density_weight(const T radius, const T smoothing_length) noexcept {
    // Return zero whenever the kernel evaluation is outside its valid domain.
    //
    // The standard SPH density kernel used here assumes:
    // - a strictly positive smoothing length
    // - a non-negative distance
    // - compact support within [0, smoothing_length]
    //
    // If any of those assumptions are violated, the kernel contributes nothing.
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }

    // Precompute h^2 once because it is reused in the support polynomial.
    const T smoothing_length_squared = smoothing_length * smoothing_length;

    // This is the Poly6-style support term:
    //     (h^2 - r^2)
    //
    // Inside the support radius, this stays non-negative and smoothly approaches
    // zero at the boundary r = h.
    const T support = smoothing_length_squared - radius * radius;

    // Normalization coefficient for the 3D density kernel:
    //     315 / (64 * pi)
    //
    // This coefficient is chosen so the kernel integrates to 1 over its support
    // in the continuous 3D setting.
    const T coeff = static_cast<T>(315.0 / (64.0 * atlas::pi));

    // Return:
    //     coeff * (h^2 - r^2)^3 / h^9
    //
    // The denominator is intentionally written as repeated multiplications rather
    // than pow(h, 9) to keep the implementation simple, explicit, and friendly to
    // host/device compilation.
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
    // Return the zero vector whenever the pressure-gradient kernel is not well-defined.
    //
    // Requirements:
    // - smoothing_length must be positive
    // - radius must be strictly positive because the formula divides by radius
    // - radius must lie inside the compact support
    //
    // The strict radius > 0 check avoids division by zero at the particle center.
    if (!(smoothing_length > T(0)) || !(radius > T(0)) || radius > smoothing_length) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // Normalization/sign coefficient for the spiky-style pressure gradient:
    //     -45 / pi
    //
    // The negative sign reflects that the kernel gradient points inward with
    // increasing radius in the scalar radial form.
    const T coeff = static_cast<T>(-45.0 / atlas::pi);

    // Radial support factor:
    //     (h - r)^2
    //
    // This smoothly goes to zero at the support boundary r = h.
    const T support = (smoothing_length - radius) * (smoothing_length - radius);

    // Return:
    //     delta * [ coeff * (h - r)^2 / (h^6 * r) ]
    //
    // Because delta is the displacement vector, dividing by radius effectively
    // turns the radial derivative into a vector-valued gradient direction.
    //
    // As above, h^6 is expanded explicitly rather than using pow().
    return delta * (coeff * support
                    / (smoothing_length * smoothing_length * smoothing_length
                       * smoothing_length * smoothing_length * smoothing_length
                       * radius));
}

template <typename T>
T
StandardSphKernel<T>::viscosity_laplacian(const T radius, const T smoothing_length) noexcept {
    // Return zero whenever the viscosity Laplacian is outside its valid domain.
    //
    // Requirements:
    // - smoothing_length must be positive
    // - radius must be non-negative
    // - radius must lie inside the compact support
    if (!(smoothing_length > T(0)) || !(radius >= T(0)) || radius > smoothing_length) {
        return T(0);
    }

    // Normalization coefficient for the standard viscosity kernel Laplacian:
    //     45 / pi
    const T coeff = static_cast<T>(45.0 / atlas::pi);

    // Return:
    //     coeff * (h - r) / h^6
    //
    // This quantity is commonly used in SPH viscosity-force formulations, where
    // the Laplacian term modulates velocity diffusion between neighboring particles.
    return coeff * (smoothing_length - radius)
        / (smoothing_length * smoothing_length * smoothing_length
           * smoothing_length * smoothing_length * smoothing_length);
}

} // namespace atlas::system