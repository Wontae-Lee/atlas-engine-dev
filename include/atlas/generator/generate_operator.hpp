#pragma once

namespace atlas::fluid {

template <typename T>
GenerateOperator<T>::GenerateOperator() noexcept
    : type(GenerateType::uniform) {

    // Initialize the discriminator to the uniform generator type.
    //
    // This guarantees that the object starts in a valid state with a well-defined
    // active union member even when the caller uses the default constructor.

    // Construct the `uniform` union member in place.
    //
    // Placement new is required because the generator implementations live inside
    // a union, so the active member must be explicitly constructed.
    new (&uniform) UniformGenerateOperator<T> {};
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const GenerateType type,
                                      const unsigned int seed) noexcept
    : type(type) {

    // Construct the union member that matches the requested generation type.
    //
    // Each branch explicitly activates one union alternative and seeds the
    // underlying generator implementation with the provided seed value.
    switch (type) {
    case GenerateType::uniform:
        // Activate the uniform generator variant.
        new (&uniform) UniformGenerateOperator<T>(seed);
        return;

    case GenerateType::maxwell_sigma:
        // Activate the Maxwell-sigma generator variant.
        new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(seed);
        return;

    case GenerateType::maxwell_boltzmann:
        // Activate the Maxwell-Boltzmann generator variant.
        new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(seed);
        return;

    default:
        // Defensively fall back to the uniform generator when the input type
        // is unknown or unsupported.
        //
        // This keeps the object valid and ensures the active union member
        // always matches the stored discriminator.
        this->type = GenerateType::uniform;
        new (&uniform) UniformGenerateOperator<T>(seed);
        return;
    }
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const GenerateOperator& other) noexcept
    : type(other.type) {
    // Copy the type discriminator first so `copy_from()` knows which union
    // member must be constructed in this object.
    copy_from(other);
}

template <typename T>
GenerateOperator<T>&
GenerateOperator<T>::operator=(const GenerateOperator& other) noexcept {
    // Guard against self-assignment so we do not destroy and reconstruct the
    // currently active union member unnecessarily.
    if (this == &other) return *this;

    // Destroy the currently active union member before reconstructing a new one.
    //
    // This is required because unions do not manage lifetime automatically for
    // non-trivial members.
    destroy_active();

    // Copy the discriminator from the source object so the correct union member
    // can be reconstructed below.
    type = other.type;

    // Copy-construct the newly active union member from the source object.
    copy_from(other);
    return *this;
}

template <typename T>
GenerateOperator<T>::~GenerateOperator() noexcept {
    // Explicitly destroy the currently active union member.
    //
    // Since the object stores non-trivial generator implementations in a union,
    // the destructor must manually dispatch to the correct member destructor.
    destroy_active();
}

template <typename T>
void
GenerateOperator<T>::destroy_active() noexcept {

    // Destroy only the currently active union member selected by `type`.
    //
    // Destroying any inactive member would be undefined behavior, so the
    // discriminator must always stay synchronized with the constructed member.
    switch (type) {
    case GenerateType::uniform:
        // Destroy the active uniform generator.
        uniform.~UniformGenerateOperator<T>();
        return;

    case GenerateType::maxwell_sigma:
        // Destroy the active Maxwell-sigma generator.
        maxwell_sigma.~MaxwellSigmaGenerateOperator<T>();
        return;

    case GenerateType::maxwell_boltzmann:
        // Destroy the active Maxwell-Boltzmann generator.
        maxwell_boltzmann.~MaxwellBoltzmannGenerateOperator<T>();
        return;

    default:
        // Defensive fallback: treat the object as if the uniform member were active.
        //
        // This preserves best-effort cleanup behavior if the discriminator ever
        // reaches an unexpected value.
        uniform.~UniformGenerateOperator<T>();
        return;
    }
}

template <typename T>
void
GenerateOperator<T>::copy_from(const GenerateOperator& other) noexcept {

    // Copy-construct the union member indicated by the current `type`.
    //
    // This function assumes that:
    // - the destination currently has no active member, or
    // - any previously active member has already been destroyed.
    switch (type) {
    case GenerateType::uniform:
        // Reconstruct the uniform generator from the source object's uniform member.
        new (&uniform) UniformGenerateOperator<T>(other.uniform);
        return;

    case GenerateType::maxwell_sigma:
        // Reconstruct the Maxwell-sigma generator from the source object.
        new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(other.maxwell_sigma);
        return;

    case GenerateType::maxwell_boltzmann:
        // Reconstruct the Maxwell-Boltzmann generator from the source object.
        new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(other.maxwell_boltzmann);
        return;

    default:
        // Defensive fallback: normalize the discriminator and reconstruct a
        // uniform generator instead.
        type = GenerateType::uniform;
        new (&uniform) UniformGenerateOperator<T>(other.uniform);
        return;
    }
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const UniformGenerateOperator<T>& op)
    : type(GenerateType::uniform) {
    // Construct this wrapper directly from an already prepared uniform generator.
    new (&uniform) UniformGenerateOperator<T>(op);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const MaxwellSigmaGenerateOperator<T>& op)
    : type(GenerateType::maxwell_sigma) {
    // Construct this wrapper directly from an already prepared Maxwell-sigma generator.
    new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(op);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const MaxwellBoltzmannGenerateOperator<T>& op)
    : type(GenerateType::maxwell_boltzmann) {
    // Construct this wrapper directly from an already prepared Maxwell-Boltzmann generator.
    new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(op);
}

template <typename T>
Vector3<T>
GenerateOperator<T>::generate(const T param0,
                              const T param1) const {

    // Dispatch generation to the currently active generator implementation.
    //
    // The meaning of `param0` and `param1` depends on the selected generation law:
    // - uniform            : uses both parameters
    // - maxwell_sigma      : uses only `param0`
    // - maxwell_boltzmann  : uses both parameters
    switch (type) {
    case GenerateType::uniform:
        // Forward both parameters to the uniform generator.
        return uniform.generate(param0, param1);

    case GenerateType::maxwell_sigma:
        // Forward only the first parameter because this generator variant
        // expects a single effective input.
        return maxwell_sigma.generate(param0);

    case GenerateType::maxwell_boltzmann:
        // Forward both parameters to the Maxwell-Boltzmann generator.
        return maxwell_boltzmann.generate(param0, param1);

    default:
        // Defensive fallback for unknown discriminator values.
        //
        // Return the zero vector rather than attempting to call an invalid
        // union member.
        return Vector3<T>(T(0), T(0), T(0));
    }
}

} // namespace atlas::fluid