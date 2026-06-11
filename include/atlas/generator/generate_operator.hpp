#pragma once
namespace atlas::fluid {
template <typename T>
GenerateOperator<T>::GenerateOperator() noexcept
    : type(GenerateType::uniform) {
    new (&uniform) UniformGenerateOperator<T> {};
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const GenerateType type,
                                      const unsigned int seed) noexcept
    : type(type) {
    switch (type) {
    case GenerateType::uniform:
        new (&uniform) UniformGenerateOperator<T>(seed);
        return;
    case GenerateType::jittering:
        new (&jittering) JitteringGenerateOperator<T>(seed);
        return;
    case GenerateType::maxwell_sigma:
        new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(seed);
        return;
    case GenerateType::maxwell_boltzmann:
        new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(seed);
        return;
    default:
        this->type = GenerateType::uniform;
        new (&uniform) UniformGenerateOperator<T>(seed);
        return;
    }
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const GenerateOperator& other) noexcept
    : type(other.type) {
    copy_from(other);
}

template <typename T>
GenerateOperator<T>&
GenerateOperator<T>::operator=(const GenerateOperator& other) noexcept {
    if (this == &other) return *this;
    destroy_active();
    type = other.type;
    copy_from(other);
    return *this;
}

template <typename T>
GenerateOperator<T>::~GenerateOperator() noexcept {
    destroy_active();
}

template <typename T>
void
GenerateOperator<T>::destroy_active() noexcept {
    switch (type) {
    case GenerateType::uniform:
        uniform.~UniformGenerateOperator<T>();
        return;
    case GenerateType::jittering:
        jittering.~JitteringGenerateOperator<T>();
        return;
    case GenerateType::maxwell_sigma:
        maxwell_sigma.~MaxwellSigmaGenerateOperator<T>();
        return;
    case GenerateType::maxwell_boltzmann:
        maxwell_boltzmann.~MaxwellBoltzmannGenerateOperator<T>();
        return;
    default:
        uniform.~UniformGenerateOperator<T>();
        return;
    }
}

template <typename T>
void
GenerateOperator<T>::copy_from(const GenerateOperator& other) noexcept {
    switch (type) {
    case GenerateType::uniform:
        new (&uniform) UniformGenerateOperator<T>(other.uniform);
        return;
    case GenerateType::jittering:
        new (&jittering) JitteringGenerateOperator<T>(other.jittering);
        return;
    case GenerateType::maxwell_sigma:
        new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(other.maxwell_sigma);
        return;
    case GenerateType::maxwell_boltzmann:
        new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(other.maxwell_boltzmann);
        return;
    default:
        type = GenerateType::uniform;
        new (&uniform) UniformGenerateOperator<T>(other.uniform);
        return;
    }
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const UniformGenerateOperator<T>& op)
    : type(GenerateType::uniform) {
    new (&uniform) UniformGenerateOperator<T>(op);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const JitteringGenerateOperator<T>& op)
    : type(GenerateType::jittering) {
    new (&jittering) JitteringGenerateOperator<T>(op);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const MaxwellSigmaGenerateOperator<T>& op)
    : type(GenerateType::maxwell_sigma) {
    new (&maxwell_sigma) MaxwellSigmaGenerateOperator<T>(op);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const MaxwellBoltzmannGenerateOperator<T>& op)
    : type(GenerateType::maxwell_boltzmann) {
    new (&maxwell_boltzmann) MaxwellBoltzmannGenerateOperator<T>(op);
}

template <typename T>
Vector3<T>
GenerateOperator<T>::generate(const T param0,
                              const T param1) const {
    switch (type) {
    case GenerateType::uniform:
        return uniform.generate(param0, param1);
    case GenerateType::jittering:
        return jittering.generate(param0, param1);
    case GenerateType::maxwell_sigma:
        return maxwell_sigma.generate(param0);
    case GenerateType::maxwell_boltzmann:
        return maxwell_boltzmann.generate(param0, param1);
    default:
        return Vector3<T>(T(0), T(0), T(0));
    }
}

template <typename T>
Vector3<T>
GenerateOperator<T>::generate(const unsigned int seed,
                              const T param0,
                              const T param1) const {
    switch (type) {
    case GenerateType::uniform:
        return uniform.generate(seed, param0, param1);
    case GenerateType::jittering:
        return jittering.generate(seed, param0, param1);
    case GenerateType::maxwell_sigma:
        return maxwell_sigma.generate(seed, param0);
    case GenerateType::maxwell_boltzmann:
        return maxwell_boltzmann.generate(seed, param0, param1);
    default:
        return Vector3<T>(T(0), T(0), T(0));
    }
}

template <typename T>
void
GenerateOperator<T>::reseed(const unsigned int seed) noexcept {
    switch (type) {
    case GenerateType::uniform:
        uniform.seed   = seed;
        uniform.engine = atlas::default_random_engine<T>(seed);
        return;
    case GenerateType::jittering:
        jittering.seed   = seed;
        jittering.engine = atlas::default_random_engine<T>(seed);
        return;
    case GenerateType::maxwell_sigma:
        maxwell_sigma.seed   = seed;
        maxwell_sigma.engine = atlas::default_random_engine<T>(seed);
        return;
    case GenerateType::maxwell_boltzmann:
        maxwell_boltzmann.seed   = seed;
        maxwell_boltzmann.engine = atlas::default_random_engine<T>(seed);
        return;
    default:
        return;
    }
}

}
