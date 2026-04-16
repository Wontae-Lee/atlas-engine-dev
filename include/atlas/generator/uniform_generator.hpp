#pragma once

#include <atlas/logging/logging.h>
#include <atlas/random/uniform_real_distribution.h>

#include <stdexcept>

namespace atlas::fluid {

template <typename T>
UniformGenerateOperator<T>::UniformGenerateOperator(const unsigned int seed) noexcept
    : seed(seed)
    , engine(seed) {
}

template <typename T>
Vector3<T>
UniformGenerateOperator<T>::generate(const T min_value,
                                     const T max_value) const {

    atlas::uniform_real_distribution<T> dist(min_value, max_value);

    return Vector3<T>(
        dist(engine),
        dist(engine),
        dist(engine));
}

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
    , _seed(seed)
    , _operator(atlas::make_host_shared<GenerateOperator<T>>(UniformGenerateOperator<T>(seed))) {
}

template <typename T>
Vector3<T>
UniformGenerator<T>::generate() const {

    return _operator->generate(_min_value, _max_value);
}

template <typename T>
const GenerateOperator<T>&
UniformGenerator<T>::generate_operator() const noexcept {

    return *_operator;
}

template <typename T>
GenerateOperator<T>
UniformGenerator<T>::make_generate_operator() const noexcept {

    return *_operator;
}

template <typename T>
T
UniformGenerator<T>::param0() const noexcept {

    return _min_value;
}

template <typename T>
T
UniformGenerator<T>::param1() const noexcept {

    return _max_value;
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

}