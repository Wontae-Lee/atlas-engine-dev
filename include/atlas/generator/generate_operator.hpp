#pragma once

namespace atlas::fluid {

template <typename T>
GenerateOperator<T>::GenerateOperator() noexcept
    : type(GenerateType::uniform) {

    // Default-initialize the tagged union as a uniform generator.
    //
    // Because the active member is managed manually, placement new is used to
    // construct the selected generator object inside the union storage.
    new (&uniform) UniformGenerateOperator<T> {};
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const GenerateType type,
                                      const unsigned int seed) noexcept
    : type(type) {

    // Construct the union member corresponding to the requested generator type.
    //
    // The runtime tag `type` determines which member becomes active.
    // Each generator is initialized with the provided seed so downstream
    // sampling behavior can remain deterministic/reproducible.
    switch (type) {
    case GenerateType::uniform:

        new (&uniform) UniformGenerateOperator<T>(seed);
        return;

    case GenerateType::maxwell_sigma:

        new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(seed);
        return;

    case GenerateType::maxwell_boltzmann:

        new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(seed);
        return;

    default:

        // Defensive fallback for unexpected tags.
        //
        // Reset the runtime tag to a valid default and construct the matching
        // union member so the object remains in a consistent state.
        this->type = GenerateType::uniform;
        new (&uniform) UniformGenerateOperator<T>(seed);
        return;
    }
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const GenerateOperator& other) noexcept
    : type(other.type) {

    // Copy-construct the currently active generator from the source object.
    //
    // The runtime tag is copied first so copy_from() knows which union member
    // must be constructed in this object.
    copy_from(other);
}

template <typename T>
GenerateOperator<T>&
GenerateOperator<T>::operator=(const GenerateOperator& other) noexcept {

    // Self-assignment requires no work.
    if (this == &other) return *this;

    // Destroy the currently active union member before constructing a new one.
    //
    // Since this type manages a tagged union manually, assignment must perform:
    // 1. destruction of the current active member
    // 2. copy of the runtime tag
    // 3. construction of the new active member from `other`
    destroy_active();

    type = other.type;

    copy_from(other);
    return *this;
}

template <typename T>
GenerateOperator<T>::~GenerateOperator() noexcept {

    // Explicitly destroy the active union member.
    //
    // The union stores non-trivial types, so lifetime management is handled
    // manually rather than automatically.
    destroy_active();
}

template <typename T>
void
GenerateOperator<T>::destroy_active() noexcept {

    // Destroy whichever generator object is currently active in the union.
    //
    // The runtime tag determines which destructor must be invoked.
    switch (type) {
    case GenerateType::uniform:

        uniform.~UniformGenerateOperator<T>();
        return;

    case GenerateType::maxwell_sigma:

        maxwell_sigma.~MaxwellSigmaGenerateOperator<T>();
        return;

    case GenerateType::maxwell_boltzmann:

        maxwell_boltzmann.~MaxwellBoltzmannGenerateOperator<T>();
        return;

    default:

        // Defensive fallback.
        //
        // If the tag is somehow invalid, destroy the uniform member as the
        // safest default assumption used elsewhere in this type.
        uniform.~UniformGenerateOperator<T>();
        return;
    }
}

template <typename T>
void
GenerateOperator<T>::copy_from(const GenerateOperator& other) noexcept {

    // Construct the active union member from the corresponding member in `other`.
    //
    // This function assumes:
    // - `type` has already been set correctly for `*this`
    // - any previously active member has already been destroyed
    switch (type) {
    case GenerateType::uniform:

        new (&uniform) UniformGenerateOperator<T>(other.uniform);
        return;

    case GenerateType::maxwell_sigma:

        new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(other.maxwell_sigma);
        return;

    case GenerateType::maxwell_boltzmann:

        new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(other.maxwell_boltzmann);
        return;

    default:

        // Defensive fallback to a valid uniform state.
        //
        // The runtime tag is normalized before construction so this object
        // remains internally consistent even if an unexpected tag appears.
        type = GenerateType::uniform;
        new (&uniform) UniformGenerateOperator<T>(other.uniform);
        return;
    }
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const UniformGenerateOperator<T>& op)
    : type(GenerateType::uniform) {

    // Construct this tagged union directly from a concrete uniform generator.
    new (&uniform) UniformGenerateOperator<T>(op);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const MaxwellSigmaGenerateOperator<T>& op)
    : type(GenerateType::maxwell_sigma) {

    // Construct this tagged union directly from a concrete Maxwell-sigma generator.
    new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(op);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const MaxwellBoltzmannGenerateOperator<T>& op)
    : type(GenerateType::maxwell_boltzmann) {

    // Construct this tagged union directly from a concrete Maxwell-Boltzmann generator.
    new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(op);
}

template <typename T>
Vector3<T>
GenerateOperator<T>::generate(const T param0,
                              const T param1) const {

    // Dispatch generation to the currently active operator.
    //
    // Different generator types interpret the input parameters differently:
    // - uniform            : uses (param0, param1)
    // - maxwell_sigma      : uses only param0
    // - maxwell_boltzmann  : uses (param0, param1)
    //
    // The runtime tag selects the appropriate generation strategy.
    switch (type) {
    case GenerateType::uniform:

        return uniform.generate(param0, param1);

    case GenerateType::maxwell_sigma:

        return maxwell_sigma.generate(param0);

    case GenerateType::maxwell_boltzmann:

        return maxwell_boltzmann.generate(param0, param1);

    default:

        // Defensive fallback for an invalid tag.
        //
        // Return the zero vector rather than invoking an invalid union member.
        return Vector3<T>(T(0), T(0), T(0));
    }
}

}