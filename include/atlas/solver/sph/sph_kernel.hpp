#pragma once
namespace atlas::system {
template <typename T>
SphKernel<T>::SphKernel() noexcept
    : type(SphKernelType::standard) {
    new (&standard) StandardSphKernel<T> {};
}

template <typename T>
SphKernel<T>::SphKernel(const SphKernelType type) noexcept
    : type(type) {
    switch (type) {
    case SphKernelType::standard:
        new (&standard) StandardSphKernel<T> {};
        return;
    case SphKernelType::cubic_spline:
        new (&cubic_spline) CubicSplineSphKernel<T> {};
        return;
    case SphKernelType::wendland_quintic:
        new (&wendland_quintic) WendlandQuinticSphKernel<T> {};
        return;
    default:
        this->type = SphKernelType::standard;
        new (&standard) StandardSphKernel<T> {};
        return;
    }
}

template <typename T>
SphKernel<T>::SphKernel(const SphKernel& other) noexcept
    : type(other.type) {
    copy_from(other);
}

template <typename T>
SphKernel<T>&
SphKernel<T>::operator=(const SphKernel& other) noexcept {
    if (this == &other) return *this;
    destroy_active();
    type = other.type;
    copy_from(other);
    return *this;
}

template <typename T>
SphKernel<T>::~SphKernel() noexcept {
    destroy_active();
}

template <typename T>
void
SphKernel<T>::destroy_active() noexcept {
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
        standard.~StandardSphKernel<T>();
        return;
    }
}

template <typename T>
void
SphKernel<T>::copy_from(const SphKernel& other) noexcept {
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
        type = SphKernelType::standard;
        new (&standard) StandardSphKernel<T>(other.standard);
        return;
    }
}

template <typename T>
SphKernel<T>::SphKernel(const StandardSphKernel<T>& op)
    : type(SphKernelType::standard) {
    new (&standard) StandardSphKernel<T>(op);
}

template <typename T>
SphKernel<T>::SphKernel(const CubicSplineSphKernel<T>& op)
    : type(SphKernelType::cubic_spline) {
    new (&cubic_spline) CubicSplineSphKernel<T>(op);
}

template <typename T>
SphKernel<T>::SphKernel(const WendlandQuinticSphKernel<T>& op)
    : type(SphKernelType::wendland_quintic) {
    new (&wendland_quintic) WendlandQuinticSphKernel<T>(op);
}

template <typename T>
T
SphKernel<T>::density_weight(const SphKernelType type,
                             const T radius,
                             const T smoothing_length) noexcept {
    switch (type) {
    case SphKernelType::standard:
        return StandardSphKernel<T>::density_weight(radius, smoothing_length);
    case SphKernelType::cubic_spline:
        return CubicSplineSphKernel<T>::density_weight(radius, smoothing_length);
    case SphKernelType::wendland_quintic:
        return WendlandQuinticSphKernel<T>::density_weight(radius, smoothing_length);
    default:
        return T(0);
    }
}

template <typename T>
Vector3<T>
SphKernel<T>::pressure_gradient(const SphKernelType type,
                                const Vector3<T>& delta,
                                const T radius,
                                const T smoothing_length) noexcept {
    switch (type) {
    case SphKernelType::standard:
        return StandardSphKernel<T>::pressure_gradient(delta, radius, smoothing_length);
    case SphKernelType::cubic_spline:
        return CubicSplineSphKernel<T>::pressure_gradient(delta, radius, smoothing_length);
    case SphKernelType::wendland_quintic:
        return WendlandQuinticSphKernel<T>::pressure_gradient(delta, radius, smoothing_length);
    default:
        return Vector3<T>(T(0), T(0), T(0));
    }
}

template <typename T>
T
SphKernel<T>::viscosity_laplacian(const SphKernelType type,
                                  const T radius,
                                  const T smoothing_length) noexcept {
    switch (type) {
    case SphKernelType::standard:
        return StandardSphKernel<T>::viscosity_laplacian(radius, smoothing_length);
    case SphKernelType::cubic_spline:
        return CubicSplineSphKernel<T>::viscosity_laplacian(radius, smoothing_length);
    case SphKernelType::wendland_quintic:
        return WendlandQuinticSphKernel<T>::viscosity_laplacian(radius, smoothing_length);
    default:
        return T(0);
    }
}

template <typename T>
T
SphKernel<T>::density_weight(const T radius, const T smoothing_length) const noexcept {
    return density_weight(type, radius, smoothing_length);
}

template <typename T>
Vector3<T>
SphKernel<T>::pressure_gradient(const Vector3<T>& delta,
                                const T radius,
                                const T smoothing_length) const noexcept {
    return pressure_gradient(type, delta, radius, smoothing_length);
}

template <typename T>
T
SphKernel<T>::viscosity_laplacian(const T radius, const T smoothing_length) const noexcept {
    return viscosity_laplacian(type, radius, smoothing_length);
}

}