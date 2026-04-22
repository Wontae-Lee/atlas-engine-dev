#pragma once

namespace atlas::system {

template <typename T>
SphKernel<T>::SphKernel() noexcept
    // Default-construct the wrapper as the "standard" SPH kernel.
    //
    // Because this type stores kernel objects inside a union, C++ does not
    // automatically know which union member is active. We therefore:
    // 1. set the runtime tag to SphKernelType::standard
    // 2. explicitly construct the matching union member in-place
    //
    // `new (&standard) ...` is placement new:
    // it constructs the object directly inside the already reserved union storage
    // instead of allocating new memory on the heap.
    : type(SphKernelType::standard) {
    new (&standard) StandardSphKernel<T> {};
}

template <typename T>
SphKernel<T>::SphKernel(const SphKernelType type) noexcept
    // Construct the wrapper from an explicit runtime kernel type.
    //
    // The wrapper must ensure that exactly one union member becomes active, and
    // that the active member matches the runtime tag stored in `type`.
    : type(type) {
    switch (type) {
    case SphKernelType::standard:
        // Activate the standard kernel in union storage.
        new (&standard) StandardSphKernel<T> {};
        return;

    case SphKernelType::cubic_spline:
        // Activate the cubic spline kernel in union storage.
        new (&cubic_spline) CubicSplineSphKernel<T> {};
        return;

    case SphKernelType::wendland_quintic:
        // Activate the Wendland quintic kernel in union storage.
        new (&wendland_quintic) WendlandQuinticSphKernel<T> {};
        return;

    default:
        // Defensive fallback:
        // if an invalid tag somehow arrives, recover into a safe default state.
        //
        // This guarantees that the wrapper still contains a valid active union
        // member and that later destruction / dispatch logic remains well-defined.
        this->type = SphKernelType::standard;
        new (&standard) StandardSphKernel<T> {};
        return;
    }
}

template <typename T>
SphKernel<T>::SphKernel(const SphKernel& other) noexcept
    // Copy-construction of a tagged union requires two things:
    // 1. copy the runtime tag
    // 2. explicitly construct the matching union member
    //
    // We copy the tag first, then delegate actual union-member construction to
    // copy_from(other), which performs placement new on the correct member.
    : type(other.type) {
    copy_from(other);
}

template <typename T>
SphKernel<T>&
SphKernel<T>::operator=(const SphKernel& other) noexcept {
    // Self-assignment guard.
    //
    // Without this check, we would destroy our active union member and then try
    // to copy from the same object, which would be unnecessary and potentially
    // fragile if the implementation changed later.
    if (this == &other) return *this;

    // The currently active union member must be destroyed before another one is
    // constructed in the same storage.
    destroy_active();

    // Copy the runtime tag first so copy_from() knows which union member should
    // be reconstructed.
    type = other.type;

    // Reconstruct the matching active union member from `other`.
    copy_from(other);
    return *this;
}

template <typename T>
SphKernel<T>::~SphKernel() noexcept {
    // Union members with non-trivial lifetime must be destroyed manually.
    //
    // The tag tells us which concrete kernel is currently active, and
    // destroy_active() dispatches to the matching destructor.
    destroy_active();
}

template <typename T>
void
SphKernel<T>::destroy_active() noexcept {
    // Manually destroy the active union member according to the runtime tag.
    //
    // This is necessary because unions do not automatically track or destroy
    // the currently active non-trivial object for us.
    switch (type) {
    case SphKernelType::standard:
        standard.~StandardSphKernel<T>();
        return;

    case SphKernelType::cubic_spline:
        cubic_spline.~CubicSplineSphKernel<T>();
        return;

    case SphKernelType::wendland_quintic:
        wendland_quintic.~WendlandQuinticSphKernel<T>();
        return;

    default:
        // Defensive fallback:
        // if the tag is invalid, destroy as standard to preserve a predictable
        // recovery path. In well-formed usage this branch should not be reached.
        standard.~StandardSphKernel<T>();
        return;
    }
}

template <typename T>
void
SphKernel<T>::copy_from(const SphKernel& other) noexcept {
    // Reconstruct the correct union member using placement new.
    //
    // This function assumes that:
    // - `type` already contains the intended destination tag
    // - the previous active member has already been destroyed if necessary
    //
    // The goal is to create an active union member in *this* that matches the
    // tag stored in `type`, copying the corresponding concrete kernel from `other`.
    switch (type) {
    case SphKernelType::standard:
        new (&standard) StandardSphKernel<T>(other.standard);
        return;

    case SphKernelType::cubic_spline:
        new (&cubic_spline) CubicSplineSphKernel<T>(other.cubic_spline);
        return;

    case SphKernelType::wendland_quintic:
        new (&wendland_quintic) WendlandQuinticSphKernel<T>(other.wendland_quintic);
        return;

    default:
        // Defensive fallback:
        // if the destination tag is invalid, normalize it to "standard" and copy
        // the standard kernel representation.
        type = SphKernelType::standard;
        new (&standard) StandardSphKernel<T>(other.standard);
        return;
    }
}

template <typename T>
SphKernel<T>::SphKernel(const StandardSphKernel<T>& op)
    // Construct the wrapper directly from an already-formed standard kernel.
    //
    // The tag is set explicitly, and the object is copied into the union storage
    // using placement new.
    : type(SphKernelType::standard) {
    new (&standard) StandardSphKernel<T>(op);
}

template <typename T>
SphKernel<T>::SphKernel(const CubicSplineSphKernel<T>& op)
    // Construct the wrapper directly from an already-formed cubic spline kernel.
    : type(SphKernelType::cubic_spline) {
    new (&cubic_spline) CubicSplineSphKernel<T>(op);
}

template <typename T>
SphKernel<T>::SphKernel(const WendlandQuinticSphKernel<T>& op)
    // Construct the wrapper directly from an already-formed Wendland quintic kernel.
    : type(SphKernelType::wendland_quintic) {
    new (&wendland_quintic) WendlandQuinticSphKernel<T>(op);
}

template <typename T>
T
SphKernel<T>::density_weight(const SphKernelType type,
                             const T radius,
                             const T smoothing_length) noexcept {
    // Static dispatch helper:
    // choose a kernel implementation from an explicit runtime tag without
    // requiring a previously constructed wrapper object.
    //
    // This is useful in places where only the kernel type is known and we want a
    // lightweight functional dispatch for density accumulation.
    switch (type) {
    case SphKernelType::standard:
        return StandardSphKernel<T>::density_weight(radius, smoothing_length);

    case SphKernelType::cubic_spline:
        return CubicSplineSphKernel<T>::density_weight(radius, smoothing_length);

    case SphKernelType::wendland_quintic:
        return WendlandQuinticSphKernel<T>::density_weight(radius, smoothing_length);

    default:
        // Invalid type fallback: return a neutral zero contribution.
        return T(0);
    }
}

template <typename T>
Vector3<T>
SphKernel<T>::pressure_gradient(const SphKernelType type,
                                const Vector3<T>& delta,
                                const T radius,
                                const T smoothing_length) noexcept {
    // Static dispatch helper for pressure-gradient evaluation.
    //
    // The selected kernel determines how the gradient of the smoothing function
    // is evaluated for the given displacement vector and smoothing length.
    switch (type) {
    case SphKernelType::standard:
        return StandardSphKernel<T>::pressure_gradient(delta, radius, smoothing_length);

    case SphKernelType::cubic_spline:
        return CubicSplineSphKernel<T>::pressure_gradient(delta, radius, smoothing_length);

    case SphKernelType::wendland_quintic:
        return WendlandQuinticSphKernel<T>::pressure_gradient(delta, radius, smoothing_length);

    default:
        // Invalid type fallback: return the zero vector so no pressure force is
        // contributed by this evaluation.
        return Vector3<T>(T(0), T(0), T(0));
    }
}

template <typename T>
T
SphKernel<T>::viscosity_laplacian(const SphKernelType type,
                                  const T radius,
                                  const T smoothing_length) noexcept {
    // Static dispatch helper for viscosity-Laplacian evaluation.
    //
    // This is typically used in viscosity-force accumulation, where the scalar
    // Laplacian of the smoothing kernel appears in a diffusion-like term.
    switch (type) {
    case SphKernelType::standard:
        return StandardSphKernel<T>::viscosity_laplacian(radius, smoothing_length);

    case SphKernelType::cubic_spline:
        return CubicSplineSphKernel<T>::viscosity_laplacian(radius, smoothing_length);

    case SphKernelType::wendland_quintic:
        return WendlandQuinticSphKernel<T>::viscosity_laplacian(radius, smoothing_length);

    default:
        // Invalid type fallback: return zero so the viscosity contribution is neutral.
        return T(0);
    }
}

template <typename T>
T
SphKernel<T>::density_weight(const T radius, const T smoothing_length) const noexcept {
    // Instance dispatch helper:
    // reuse the static dispatch routine, but provide the currently active kernel tag
    // stored in this wrapper.
    //
    // This path is used when the caller already holds a constructed SphKernel<T>.
    return density_weight(type, radius, smoothing_length);
}

template <typename T>
Vector3<T>
SphKernel<T>::pressure_gradient(const Vector3<T>& delta,
                                const T radius,
                                const T smoothing_length) const noexcept {
    // Instance dispatch helper for pressure-gradient evaluation using the active
    // runtime kernel stored in this wrapper.
    return pressure_gradient(type, delta, radius, smoothing_length);
}

template <typename T>
T
SphKernel<T>::viscosity_laplacian(const T radius, const T smoothing_length) const noexcept {
    // Instance dispatch helper for viscosity-Laplacian evaluation using the
    // active runtime kernel stored in this wrapper.
    return viscosity_laplacian(type, radius, smoothing_length);
}

} // namespace atlas::system