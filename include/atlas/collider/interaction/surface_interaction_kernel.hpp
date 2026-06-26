#pragma once

namespace atlas {

namespace detail {

    template <typename T>
    using SurfaceInteractionVariant = DeviceVariant<
        SurfaceInteractionKernel<T>,
        SurfaceInteractionType,
        SurfaceInteractionType::isothermal,
        DeviceVariantCase<
            SurfaceInteractionKernel<T>,
            SurfaceInteractionType,
            SurfaceInteractionType::isothermal,
            IsothermalSurfaceInteraction<T>,
            &SurfaceInteractionKernel<T>::isothermal>,
        DeviceVariantCase<
            SurfaceInteractionKernel<T>,
            SurfaceInteractionType,
            SurfaceInteractionType::maxwellian,
            MaxwellianSurfaceInteraction<T>,
            &SurfaceInteractionKernel<T>::maxwellian>>;

}

template <typename T>
SurfaceInteractionKernel<T>::SurfaceInteractionKernel() noexcept {
    detail::SurfaceInteractionVariant<T>::construct(*this, SurfaceInteractionType::isothermal);
}

template <typename T>
SurfaceInteractionKernel<T>::SurfaceInteractionKernel(
    const IsothermalSurfaceInteraction<T>& interaction) noexcept {
    detail::SurfaceInteractionVariant<T>::construct_payload(*this, interaction);
}

template <typename T>
SurfaceInteractionKernel<T>::SurfaceInteractionKernel(
    const MaxwellianSurfaceInteraction<T>& interaction) noexcept {
    detail::SurfaceInteractionVariant<T>::construct_payload(*this, interaction);
}

template <typename T>
SurfaceInteractionKernel<T>::SurfaceInteractionKernel(const SurfaceInteractionKernel& other) noexcept {
    detail::SurfaceInteractionVariant<T>::copy_construct(*this, other);
}

template <typename T>
SurfaceInteractionKernel<T>&
SurfaceInteractionKernel<T>::operator=(const SurfaceInteractionKernel& other) noexcept {
    detail::SurfaceInteractionVariant<T>::assign(*this, other);
    return *this;
}

template <typename T>
SurfaceInteractionKernel<T>::~SurfaceInteractionKernel() noexcept {
    destroy_active();
}

template <typename T>
Vector3<T>
SurfaceInteractionKernel<T>::operator()(const Vector3<T>& incident,
                                        const Vector3<T>& normal) const noexcept {
    return detail::SurfaceInteractionVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& interaction) noexcept { return interaction(incident, normal); },
        Vector3<T>(T(0), T(0), T(0)));
}

template <typename T>
FluidInternalEnergy<T>
SurfaceInteractionKernel<T>::internal_energy(
    const FluidInternalEnergy<T>& incident_energy,
    const Vector3<T>& incident_velocity,
    const Vector3<T>& normal,
    const MaterialProperties<T>& material) const noexcept {
    return detail::SurfaceInteractionVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& interaction) noexcept {
            return interaction.internal_energy(incident_energy, incident_velocity, normal, material);
        },
        incident_energy);
}

template <typename T>
void
SurfaceInteractionKernel<T>::destroy_active() noexcept {
    detail::SurfaceInteractionVariant<T>::destroy(*this);
}

template <typename T>
void
SurfaceInteractionKernel<T>::copy_from(const SurfaceInteractionKernel& other) noexcept {
    detail::SurfaceInteractionVariant<T>::copy_construct(*this, other);
}

}