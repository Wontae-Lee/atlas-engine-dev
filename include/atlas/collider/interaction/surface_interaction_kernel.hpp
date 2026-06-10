#pragma once

#include <new>

namespace atlas::system {

template <typename T>
SurfaceInteractionKernel<T>::SurfaceInteractionKernel() noexcept
    : type(SurfaceInteractionType::isothermal)
    , isothermal() {
}

template <typename T>
SurfaceInteractionKernel<T>::SurfaceInteractionKernel(
    const IsothermalSurfaceInteraction<T>& interaction) noexcept
    : type(SurfaceInteractionType::isothermal)
    , isothermal(interaction) {
}

template <typename T>
SurfaceInteractionKernel<T>::SurfaceInteractionKernel(
    const MaxwellianSurfaceInteraction<T>& interaction) noexcept
    : type(SurfaceInteractionType::maxwellian)
    , maxwellian(interaction) {
}

template <typename T>
SurfaceInteractionKernel<T>::SurfaceInteractionKernel(const SurfaceInteractionKernel& other) noexcept {
    copy_from(other);
}

template <typename T>
SurfaceInteractionKernel<T>&
SurfaceInteractionKernel<T>::operator=(const SurfaceInteractionKernel& other) noexcept {
    if (this != &other) {
        destroy_active();
        copy_from(other);
    }

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
    switch (type) {
    case SurfaceInteractionType::isothermal:
        return isothermal(incident, normal);
    case SurfaceInteractionType::maxwellian:
        return maxwellian(incident, normal);
    }

    return Vector3<T>(T(0), T(0), T(0));
}

template <typename T>
fluid::FluidInternalEnergy<T>
SurfaceInteractionKernel<T>::internal_energy(
    const fluid::FluidInternalEnergy<T>& incident_energy,
    const Vector3<T>& incident_velocity,
    const Vector3<T>& normal,
    const MaterialProperties<T>& material) const noexcept {
    switch (type) {
    case SurfaceInteractionType::isothermal:
        return incident_energy;
    case SurfaceInteractionType::maxwellian:
        return maxwellian.internal_energy(incident_energy, incident_velocity, normal, material);
    }

    return incident_energy;
}

template <typename T>
void
SurfaceInteractionKernel<T>::destroy_active() noexcept {
    switch (type) {
    case SurfaceInteractionType::isothermal:
        isothermal.~IsothermalSurfaceInteraction<T>();
        break;
    case SurfaceInteractionType::maxwellian:
        maxwellian.~MaxwellianSurfaceInteraction<T>();
        break;
    }
}

template <typename T>
void
SurfaceInteractionKernel<T>::copy_from(const SurfaceInteractionKernel& other) noexcept {
    type = other.type;

    switch (type) {
    case SurfaceInteractionType::isothermal:
        new (&isothermal) IsothermalSurfaceInteraction<T>(other.isothermal);
        break;
    case SurfaceInteractionType::maxwellian:
        new (&maxwellian) MaxwellianSurfaceInteraction<T>(other.maxwellian);
        break;
    }
}

} // namespace atlas::system
