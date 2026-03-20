#pragma once

#include <atlas/logging/logging.h>

#include <stdexcept>

namespace atlas::system {

template <typename T>
typename UniformGenerator<T>::Builder
UniformGenerator<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
UniformGenerator<T>::UniformGenerator(const T min_value,
                                      const T max_value,
                                      const unsigned int seed) noexcept
    : _min_value(min_value)
    , _max_value(max_value)
    , _seed(seed) { }

template <typename T>
void
UniformGenerator<T>::generate(DeviceBuffer<Vector3<T>>& values) const {
    UniformGenerateOperator<T> {}.generate(values, _min_value, _max_value, _seed);
}

template <typename T>
GenerateType
UniformGenerator<T>::type() const noexcept {
    return GenerateType::uniform;
}

template <typename T>
UniformGenerator<T>
UniformGenerator<T>::Builder::build() const {
    validate();
    return UniformGenerator<T>(*_min_value, *_max_value, _seed);
}

template <typename T>
atlas::host_shared_ptr<UniformGenerator<T>>
UniformGenerator<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<UniformGenerator<T>>(build());
}

template <typename T>
typename UniformGenerator<T>::Builder&
UniformGenerator<T>::Builder::with_min_value(const T min_value) noexcept {
    _min_value = min_value;
    return *this;
}

template <typename T>
typename UniformGenerator<T>::Builder&
UniformGenerator<T>::Builder::with_max_value(const T max_value) noexcept {
    _max_value = max_value;
    return *this;
}

template <typename T>
typename UniformGenerator<T>::Builder&
UniformGenerator<T>::Builder::with_seed(const unsigned int seed) noexcept {
    _seed = seed;
    return *this;
}

template <typename T>
void
UniformGenerator<T>::Builder::validate() const {
    if (!_min_value.has_value() || !_max_value.has_value()) {
        atlas::logger::error()
            << "UniformGenerator::Builder: min_value and max_value must be provided.";
        throw std::runtime_error("UniformGenerator::Builder: min_value and max_value must be provided.");
    }

    if (!(*_min_value < *_max_value)) {
        atlas::logger::error()
            << "UniformGenerator::Builder: min_value must be less than max_value.";
        throw std::runtime_error("UniformGenerator::Builder: min_value must be less than max_value.");
    }
}

} // namespace atlas::system
