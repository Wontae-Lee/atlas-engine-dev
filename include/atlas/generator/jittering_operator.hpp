#pragma once
#include <atlas/random/uniform_real_distribution.h>
#include <cmath>
#include <stdexcept>
namespace atlas::fluid {
template <typename T>
JitteringGenerateOperator<T>::JitteringGenerateOperator(const unsigned int seed,
                                                        const T base_value,
                                                        const T jitter_radius) noexcept
    : seed(seed)
    , base_value(base_value)
    , jitter_radius(jitter_radius)
    , engine(seed) {
}

template <typename T>
Vector3<T>
JitteringGenerateOperator<T>::generate(const T,
                                       const T) const {
    const T radius = jitter_radius < T(0) ? -jitter_radius : jitter_radius;
    atlas::uniform_real_distribution<T> distribution(-radius, radius);
    return Vector3<T>(
        base_value + distribution(engine),
        base_value + distribution(engine),
        base_value + distribution(engine));
}

template <typename T>
typename JitteringOperator<T>::Builder
JitteringOperator<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
JitteringOperator<T>::JitteringOperator(const T base_value,
                                        const T jitter_radius,
                                        const unsigned int seed) noexcept
    : _base_value(base_value)
    , _jitter_radius(jitter_radius)
    , _seed(seed)
    , _operator(atlas::make_host_shared<GenerateOperator<T>>(JitteringGenerateOperator<T>(
          seed,
          base_value,
          jitter_radius))) {
}

template <typename T>
Vector3<T>
JitteringOperator<T>::generate() const {
    return _operator->generate(_base_value, _jitter_radius);
}

template <typename T>
const GenerateOperator<T>&
JitteringOperator<T>::generate_operator() const noexcept {
    return *_operator;
}

template <typename T>
GenerateOperator<T>
JitteringOperator<T>::make_generate_operator() const noexcept {
    return *_operator;
}

template <typename T>
T
JitteringOperator<T>::param0() const noexcept {
    return _base_value;
}

template <typename T>
T
JitteringOperator<T>::param1() const noexcept {
    return _jitter_radius;
}

template <typename T>
GenerateType
JitteringOperator<T>::type() const noexcept {
    return GenerateType::jittering;
}

template <typename T>
typename JitteringOperator<T>::Builder&
JitteringOperator<T>::Builder::with_base_value(const T base_value) noexcept {
    _base_value = base_value;
    return *this;
}

template <typename T>
typename JitteringOperator<T>::Builder&
JitteringOperator<T>::Builder::with_jitter_radius(const T jitter_radius) noexcept {
    _jitter_radius = jitter_radius;
    return *this;
}

template <typename T>
typename JitteringOperator<T>::Builder&
JitteringOperator<T>::Builder::with_seed(const unsigned int seed) noexcept {
    _seed = seed;
    return *this;
}

template <typename T>
void
JitteringOperator<T>::Builder::validate() const {
    if (!_base_value.has_value() || !_jitter_radius.has_value()) {
        throw std::runtime_error(
            "JitteringOperator::Builder: base_value and jitter_radius must be provided.");
    }
    if (!std::isfinite(static_cast<double>(*_base_value))
        || !std::isfinite(static_cast<double>(*_jitter_radius))) {
        throw std::runtime_error("JitteringOperator::Builder: parameters must be finite.");
    }
    if (!(*_jitter_radius >= T(0))) {
        throw std::runtime_error("JitteringOperator::Builder: jitter_radius must be non-negative.");
    }
}

template <typename T>
JitteringOperator<T>
JitteringOperator<T>::Builder::build() const {
    validate();
    return JitteringOperator<T>(*_base_value, *_jitter_radius, _seed);
}

template <typename T>
atlas::host_shared_ptr<JitteringOperator<T>>
JitteringOperator<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<JitteringOperator<T>>(build());
}

}