#pragma once
namespace atlas {

namespace detail {

    template <typename T>
    using SphKernelVariant = DeviceVariant<
        SphKernel<T>,
        SphKernelType,
        SphKernelType::standard,
        DeviceVariantCase<SphKernelType::standard, &SphKernel<T>::standard>,
        DeviceVariantCase<SphKernelType::cubic_spline, &SphKernel<T>::cubic_spline>,
        DeviceVariantCase<SphKernelType::wendland_quintic, &SphKernel<T>::wendland_quintic>>;

}

template <typename T>
SphKernel<T>::SphKernel() noexcept {
    detail::SphKernelVariant<T>::construct(*this, SphKernelType::standard);
}

template <typename T>
SphKernel<T>::SphKernel(const SphKernelType type) noexcept {
    detail::SphKernelVariant<T>::construct(*this, type);
}

template <typename T>
SphKernel<T>::SphKernel(const SphKernel& other) noexcept {
    detail::SphKernelVariant<T>::copy_construct(*this, other);
}

template <typename T>
SphKernel<T>&
SphKernel<T>::operator=(const SphKernel& other) noexcept {
    detail::SphKernelVariant<T>::assign(*this, other);
    return *this;
}

template <typename T>
SphKernel<T>::~SphKernel() noexcept {
    destroy_active();
}

template <typename T>
void
SphKernel<T>::destroy_active() noexcept {
    detail::SphKernelVariant<T>::destroy(*this);
}

template <typename T>
void
SphKernel<T>::copy_from(const SphKernel& other) noexcept {
    detail::SphKernelVariant<T>::copy_construct(*this, other);
}

template <typename T>
template <typename Payload, std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, SphKernel<T>>, int>>
SphKernel<T>::SphKernel(const Payload& op) {
    detail::SphKernelVariant<T>::construct_payload(*this, op);
}

template <typename T>
T
SphKernel<T>::density_weight(const SphKernelType type,
                             const T radius,
                             const T cell_size) noexcept {
    return detail::SphKernelVariant<T>::visit_type(
        type,
        [&] ATLAS_ALL_DEVICE (auto kernel_tag) noexcept {
            using Kernel = typename decltype(kernel_tag)::type;
            return Kernel::density_weight(radius, cell_size);
        },
        T(0));
}

template <typename T>
Vector3<T>
SphKernel<T>::pressure_gradient(const SphKernelType type,
                                const Vector3<T>& delta,
                                const T radius,
                                const T cell_size) noexcept {
    return detail::SphKernelVariant<T>::visit_type(
        type,
        [&] ATLAS_ALL_DEVICE (auto kernel_tag) noexcept {
            using Kernel = typename decltype(kernel_tag)::type;
            return Kernel::pressure_gradient(delta, radius, cell_size);
        },
        Vector3<T>(T(0), T(0), T(0)));
}

template <typename T>
T
SphKernel<T>::viscosity_laplacian(const SphKernelType type,
                                  const T radius,
                                  const T cell_size) noexcept {
    return detail::SphKernelVariant<T>::visit_type(
        type,
        [&] ATLAS_ALL_DEVICE (auto kernel_tag) noexcept {
            using Kernel = typename decltype(kernel_tag)::type;
            return Kernel::viscosity_laplacian(radius, cell_size);
        },
        T(0));
}

template <typename T>
T
SphKernel<T>::density_weight(const T radius, const T cell_size) const noexcept {
    return density_weight(type, radius, cell_size);
}

template <typename T>
Vector3<T>
SphKernel<T>::pressure_gradient(const Vector3<T>& delta,
                                const T radius,
                                const T cell_size) const noexcept {
    return pressure_gradient(type, delta, radius, cell_size);
}

template <typename T>
T
SphKernel<T>::viscosity_laplacian(const T radius, const T cell_size) const noexcept {
    return viscosity_laplacian(type, radius, cell_size);
}

}