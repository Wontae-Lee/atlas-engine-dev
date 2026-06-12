#pragma once
namespace atlas {

namespace detail {

    template <typename T>
    using GenerateOperatorVariant = DeviceVariant<
        GenerateOperator<T>,
        GenerateType,
        GenerateType::uniform,
        DeviceVariantCase<
            GenerateOperator<T>,
            GenerateType,
            GenerateType::uniform,
            UniformGenerateOperator<T>,
            &GenerateOperator<T>::uniform>,
        DeviceVariantCase<
            GenerateOperator<T>,
            GenerateType,
            GenerateType::jittering,
            JitteringGenerateOperator<T>,
            &GenerateOperator<T>::jittering>,
        DeviceVariantCase<
            GenerateOperator<T>,
            GenerateType,
            GenerateType::maxwell_sigma,
            MaxwellSigmaGenerateOperator<T>,
            &GenerateOperator<T>::maxwell_sigma>,
        DeviceVariantCase<
            GenerateOperator<T>,
            GenerateType,
            GenerateType::maxwell_boltzmann,
            MaxwellBoltzmannGenerateOperator<T>,
            &GenerateOperator<T>::maxwell_boltzmann>>;

}

template <typename T>
GenerateOperator<T>::GenerateOperator() noexcept {
    detail::GenerateOperatorVariant<T>::construct(*this, GenerateType::uniform);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const GenerateType type,
                                      const unsigned int seed) noexcept {
    detail::GenerateOperatorVariant<T>::construct(*this, type, seed);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const GenerateOperator& other) noexcept {
    detail::GenerateOperatorVariant<T>::copy_construct(*this, other);
}

template <typename T>
GenerateOperator<T>&
GenerateOperator<T>::operator=(const GenerateOperator& other) noexcept {
    detail::GenerateOperatorVariant<T>::assign(*this, other);
    return *this;
}

template <typename T>
GenerateOperator<T>::~GenerateOperator() noexcept {
    destroy_active();
}

template <typename T>
void
GenerateOperator<T>::destroy_active() noexcept {
    detail::GenerateOperatorVariant<T>::destroy(*this);
}

template <typename T>
void
GenerateOperator<T>::copy_from(const GenerateOperator& other) noexcept {
    detail::GenerateOperatorVariant<T>::copy_construct(*this, other);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const UniformGenerateOperator<T>& op) {
    detail::GenerateOperatorVariant<T>::construct_payload(*this, op);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const JitteringGenerateOperator<T>& op) {
    detail::GenerateOperatorVariant<T>::construct_payload(*this, op);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const MaxwellSigmaGenerateOperator<T>& op) {
    detail::GenerateOperatorVariant<T>::construct_payload(*this, op);
}

template <typename T>
GenerateOperator<T>::GenerateOperator(const MaxwellBoltzmannGenerateOperator<T>& op) {
    detail::GenerateOperatorVariant<T>::construct_payload(*this, op);
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