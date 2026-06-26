#pragma once

#include <stdexcept>

namespace atlas {

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
    , _operator(JitteringGenerateOperator<T>(seed, base_value, jitter_radius)) {
}

template <typename T>
Vector3<T>
JitteringOperator<T>::generate() const {
    return _operator.generate(_base_value, _jitter_radius);
}

template <typename T>
const GenerateOperator<T>&
JitteringOperator<T>::generate_operator() const noexcept {
    return _operator;
}

template <typename T>
GenerateOperator<T>
JitteringOperator<T>::make_generate_operator() const noexcept {
    return _operator;
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

    if (!atlas::isfinite(*_base_value)
        || !atlas::isfinite(*_jitter_radius)) {
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