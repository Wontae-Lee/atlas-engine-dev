#pragma once
namespace atlas {

namespace detail {

    template <typename T>
    using GenerateOperatorVariant = DeviceVariant<
        GenerateOperator<T>,
        GenerateType,
        GenerateType::uniform,
        DeviceVariantCase<GenerateType::uniform, &GenerateOperator<T>::uniform>,
        DeviceVariantCase<GenerateType::jittering, &GenerateOperator<T>::jittering>,
        DeviceVariantCase<GenerateType::maxwell_sigma, &GenerateOperator<T>::maxwell_sigma>,
        DeviceVariantCase<GenerateType::maxwell_boltzmann, &GenerateOperator<T>::maxwell_boltzmann>>;

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
template <typename Payload, std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, GenerateOperator<T>>, int>>
GenerateOperator<T>::GenerateOperator(const Payload& op) {
    detail::GenerateOperatorVariant<T>::construct_payload(*this, op);
}

template <typename T>
Vector3<T>
GenerateOperator<T>::generate(const T param0,
                              const T param1) const {
    return detail::GenerateOperatorVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& op) { return op.generate(param0, param1); },
        Vector3<T>(T(0), T(0), T(0)));
}

template <typename T>
Vector3<T>
GenerateOperator<T>::generate(const unsigned int seed,
                              const T param0,
                              const T param1) const {
    return detail::GenerateOperatorVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& op) { return op.generate(seed, param0, param1); },
        Vector3<T>(T(0), T(0), T(0)));
}

template <typename T>
void
GenerateOperator<T>::reseed(const unsigned int seed) noexcept {
    detail::GenerateOperatorVariant<T>::apply(
        *this,
        [&] ATLAS_ALL_DEVICE (auto& op) noexcept {
            op.seed   = seed;
            op.engine = atlas::default_random_engine<T>(seed);
        });
}

}