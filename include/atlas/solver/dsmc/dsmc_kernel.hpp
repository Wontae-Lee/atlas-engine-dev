#pragma once

#include <cmath>

namespace atlas::system {
template <typename T>
DsmcKernel<T>::DsmcKernel() noexcept
    : type(DsmcKernelType::hard_sphere) {
    new (&hard_sphere) HardSphereKernel<T> {};
}

template <typename T>
DsmcKernel<T>::DsmcKernel(const DsmcKernelType type) noexcept
    : type(type) {
    switch (type) {
    case DsmcKernelType::hard_sphere:
        new (&hard_sphere) HardSphereKernel<T> {};
        return;
    case DsmcKernelType::variable_hard_sphere:
        new (&variable_hard_sphere) VariableHardSphereKernel<T> {};
        return;
    case DsmcKernelType::variable_soft_sphere:
        new (&variable_soft_sphere) VariableSoftSphereKernel<T> {};
        return;
    default:
        this->type = DsmcKernelType::hard_sphere;
        new (&hard_sphere) HardSphereKernel<T> {};
        return;
    }
}

template <typename T>
DsmcKernel<T>::DsmcKernel(const DsmcKernel& other) noexcept
    : type(other.type) {
    copy_from(other);
}

template <typename T>
DsmcKernel<T>&
DsmcKernel<T>::operator=(const DsmcKernel& other) noexcept {
    if (this == &other) return *this;
    destroy_active();
    type = other.type;
    copy_from(other);
    return *this;
}

template <typename T>
DsmcKernel<T>::~DsmcKernel() noexcept {
    destroy_active();
}

template <typename T>
void
DsmcKernel<T>::destroy_active() noexcept {
    switch (type) {
    case DsmcKernelType::hard_sphere:
        hard_sphere.~HardSphereKernel<T>();
        return;
    case DsmcKernelType::variable_hard_sphere:
        variable_hard_sphere.~VariableHardSphereKernel<T>();
        return;
    case DsmcKernelType::variable_soft_sphere:
        variable_soft_sphere.~VariableSoftSphereKernel<T>();
        return;
    default:
        hard_sphere.~HardSphereKernel<T>();
        return;
    }
}

template <typename T>
void
DsmcKernel<T>::copy_from(const DsmcKernel& other) noexcept {
    switch (type) {
    case DsmcKernelType::hard_sphere:
        new (&hard_sphere) HardSphereKernel<T>(other.hard_sphere);
        return;
    case DsmcKernelType::variable_hard_sphere:
        new (&variable_hard_sphere) VariableHardSphereKernel<T>(other.variable_hard_sphere);
        return;
    case DsmcKernelType::variable_soft_sphere:
        new (&variable_soft_sphere) VariableSoftSphereKernel<T>(other.variable_soft_sphere);
        return;
    default:
        type = DsmcKernelType::hard_sphere;
        new (&hard_sphere) HardSphereKernel<T>(other.hard_sphere);
        return;
    }
}

template <typename T>
DsmcKernel<T>::DsmcKernel(const HardSphereKernel<T>& op)
    : type(DsmcKernelType::hard_sphere) {
    new (&hard_sphere) HardSphereKernel<T>(op);
}

template <typename T>
DsmcKernel<T>::DsmcKernel(const VariableHardSphereKernel<T>& op)
    : type(DsmcKernelType::variable_hard_sphere) {
    new (&variable_hard_sphere) VariableHardSphereKernel<T>(op);
}

template <typename T>
DsmcKernel<T>::DsmcKernel(const VariableSoftSphereKernel<T>& op)
    : type(DsmcKernelType::variable_soft_sphere) {
    new (&variable_soft_sphere) VariableSoftSphereKernel<T>(op);
}

template <typename T>
DsmcPairParameters<T>
DsmcKernel<T>::pair_parameters(const MaterialProperties<T>& lhs,
                               const MaterialProperties<T>& rhs) noexcept {
    DsmcPairParameters<T> pair {};
    const T lhs_mass = lhs.molecular_mass;
    const T rhs_mass = rhs.molecular_mass;
    const T mass_sum = lhs_mass + rhs_mass;
    if (!(lhs_mass > T(0)) || !(rhs_mass > T(0)) || !(mass_sum > T(0))) {
        return pair;
    }

    pair.reduced_mass = lhs_mass * rhs_mass / mass_sum;
    pair.viscosity_index = (lhs.viscosity_index.value_or(T(0.5))
                            + rhs.viscosity_index.value_or(T(0.5)))
        * T(0.5);
    pair.scattering_parameter = (lhs.scattering_parameter.value_or(T(1))
                                 + rhs.scattering_parameter.value_or(T(1)))
        * T(0.5);

    if (lhs.reference_diameter.has_value() && rhs.reference_diameter.has_value()) {
        pair.reference_diameter = (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * T(0.5);
    }
    if (lhs.reference_temperature.has_value() && rhs.reference_temperature.has_value()) {
        pair.reference_temperature = (lhs.reference_temperature.value() + rhs.reference_temperature.value()) * T(0.5);
    }

    pair.valid = pair.reduced_mass > T(0);
    return pair;
}

template <typename T>
T
DsmcKernel<T>::cross_section(const DsmcKernelType type,
                             const MaterialProperties<T>& lhs,
                             const MaterialProperties<T>& rhs,
                             const T relative_speed) noexcept {
    switch (type) {
    case DsmcKernelType::hard_sphere:
        return HardSphereKernel<T>::cross_section(lhs, rhs);
    case DsmcKernelType::variable_hard_sphere:
        return VariableHardSphereKernel<T>::cross_section(lhs, rhs, relative_speed);
    case DsmcKernelType::variable_soft_sphere:
        return VariableSoftSphereKernel<T>::cross_section(lhs, rhs, relative_speed);
    default:
        return T(0);
    }
}

template <typename T>
void
DsmcKernel<T>::operator()(Vector3<T>& lhs_velocity,
                          Vector3<T>& rhs_velocity,
                          const MaterialProperties<T>& lhs,
                          const MaterialProperties<T>& rhs) const noexcept {
    switch (type) {
    case DsmcKernelType::hard_sphere:
        hard_sphere(lhs_velocity, rhs_velocity, lhs, rhs);
        return;
    case DsmcKernelType::variable_hard_sphere:
        variable_hard_sphere(lhs_velocity, rhs_velocity, lhs, rhs);
        return;
    case DsmcKernelType::variable_soft_sphere:
        variable_soft_sphere(lhs_velocity, rhs_velocity, lhs, rhs);
        return;
    default:
        return;
    }
}

template <typename T>
T
DsmcKernel<T>::sigma_g(const MaterialProperties<T>* properties_ptr,
                       const std::size_t species_i,
                       const std::size_t species_j,
                       const T relative_speed_squared) const noexcept {
    if (!(relative_speed_squared > T(0))) {
        return T(0);
    }

    const T relative_speed = static_cast<T>(std::sqrt(static_cast<double>(relative_speed_squared)));
    return DsmcKernel<T>::cross_section(
               type,
               properties_ptr[species_i],
               properties_ptr[species_j],
               relative_speed)
        * relative_speed;
}


}
