#pragma once

namespace atlas::system {

template <typename T>
DsmcKernel<T>::DsmcKernel() noexcept
    : type(DsmcKernelType::hard_sphere) {
    // Construct the default active union member as a hard-sphere kernel.
    new (&hard_sphere) HardSphereKernel<T> {};
}

template <typename T>
DsmcKernel<T>::DsmcKernel(const DsmcKernelType type) noexcept
    : type(type) {
    // Construct the union member that corresponds to the requested kernel type.
    switch (type) {
    case DsmcKernelType::hard_sphere:
        // Activate the hard-sphere kernel storage.
        new (&hard_sphere) HardSphereKernel<T> {};
        return;

    case DsmcKernelType::variable_hard_sphere:
        // Activate the variable-hard-sphere kernel storage.
        new (&variable_hard_sphere) VariableHardSphereKernel<T> {};
        return;

    case DsmcKernelType::variable_soft_sphere:
        // Activate the variable-soft-sphere kernel storage.
        new (&variable_soft_sphere) VariableSoftSphereKernel<T> {};
        return;

    default:
        // Fall back to a valid default state when an unknown type is provided.
        this->type = DsmcKernelType::hard_sphere;
        new (&hard_sphere) HardSphereKernel<T> {};
        return;
    }
}

template <typename T>
DsmcKernel<T>::DsmcKernel(const DsmcKernel& other) noexcept
    : type(other.type) {
    // Copy-construct the currently active union member from the source object.
    copy_from(other);
}

template <typename T>
DsmcKernel<T>&
DsmcKernel<T>::operator=(const DsmcKernel& other) noexcept {
    if (this == &other) return *this;

    // Destroy the currently active union member before reconstructing it.
    destroy_active();

    // Update the discriminator to match the source object.
    type = other.type;

    // Copy-construct the matching union member from the source object.
    copy_from(other);
    return *this;
}

template <typename T>
DsmcKernel<T>::~DsmcKernel() noexcept {
    // Destroy the active union member explicitly because the storage is managed manually.
    destroy_active();
}

template <typename T>
void
DsmcKernel<T>::destroy_active() noexcept {
    // Explicitly destroy only the union member selected by the current discriminator.
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
        // Preserve a safe cleanup path by assuming the default hard-sphere member.
        hard_sphere.~HardSphereKernel<T>();
        return;
    }
}

template <typename T>
void
DsmcKernel<T>::copy_from(const DsmcKernel& other) noexcept {
    // Copy-construct the union member associated with the current discriminator.
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
        // Recover to the default hard-sphere representation if the discriminator is invalid.
        type = DsmcKernelType::hard_sphere;
        new (&hard_sphere) HardSphereKernel<T>(other.hard_sphere);
        return;
    }
}

template <typename T>
DsmcKernel<T>::DsmcKernel(const HardSphereKernel<T>& op)
    : type(DsmcKernelType::hard_sphere) {
    // Initialize the tagged union from an existing hard-sphere kernel object.
    new (&hard_sphere) HardSphereKernel<T>(op);
}

template <typename T>
DsmcKernel<T>::DsmcKernel(const VariableHardSphereKernel<T>& op)
    : type(DsmcKernelType::variable_hard_sphere) {
    // Initialize the tagged union from an existing variable-hard-sphere kernel object.
    new (&variable_hard_sphere) VariableHardSphereKernel<T>(op);
}

template <typename T>
DsmcKernel<T>::DsmcKernel(const VariableSoftSphereKernel<T>& op)
    : type(DsmcKernelType::variable_soft_sphere) {
    // Initialize the tagged union from an existing variable-soft-sphere kernel object.
    new (&variable_soft_sphere) VariableSoftSphereKernel<T>(op);
}

template <typename T>
T
DsmcKernel<T>::cross_section(const DsmcKernelType type,
                             const MatrialProperties<T>& lhs,
                             const MatrialProperties<T>& rhs,
                             const T relative_speed) noexcept {
    // Dispatch the cross-section evaluation to the kernel implementation selected by the tag.
    switch (type) {
    case DsmcKernelType::hard_sphere:
        return HardSphereKernel<T>::cross_section(lhs, rhs);

    case DsmcKernelType::variable_hard_sphere:
        return VariableHardSphereKernel<T>::cross_section(lhs, rhs, relative_speed);

    case DsmcKernelType::variable_soft_sphere:
        return VariableSoftSphereKernel<T>::cross_section(lhs, rhs, relative_speed);

    default:
        // Return zero when no valid kernel type is available.
        return T(0);
    }
}

template <typename T>
void
DsmcKernel<T>::operator()(Vector3<T>& lhs_velocity,
                          Vector3<T>& rhs_velocity,
                          const MatrialProperties<T>& lhs,
                          const MatrialProperties<T>& rhs) const noexcept {
    // Dispatch the collision update to the currently active kernel implementation.
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
        // Do nothing when the discriminator does not identify a valid active kernel.
        return;
    }
}

} // namespace atlas::system