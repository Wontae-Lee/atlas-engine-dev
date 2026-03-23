#pragma once

#include <atlas/logging/logging.h>

#include <stdexcept>

namespace atlas::system {

template <typename T>
typename MaxwellSigmaGenerator<T>::Builder
MaxwellSigmaGenerator<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
MaxwellSigmaGenerator<T>::MaxwellSigmaGenerator(const T sigma,
                                                const unsigned int seed) noexcept
    : _sigma(sigma)
    , _seed(seed)
    , _operator(seed) { }

template <typename T>
Vector3<T>
MaxwellSigmaGenerator<T>::generate() const {
    return _operator.generate(_sigma);
}

template <typename T>
GenerateType
MaxwellSigmaGenerator<T>::type() const noexcept {
    return GenerateType::maxwell_sigma;
}

template <typename T>
MaxwellSigmaGenerator<T>
MaxwellSigmaGenerator<T>::Builder::build() const {
    validate();
    return MaxwellSigmaGenerator<T>(*_sigma, _seed);
}

template <typename T>
atlas::host_shared_ptr<MaxwellSigmaGenerator<T>>
MaxwellSigmaGenerator<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<MaxwellSigmaGenerator<T>>(build());
}

template <typename T>
typename MaxwellSigmaGenerator<T>::Builder&
MaxwellSigmaGenerator<T>::Builder::with_sigma(const T sigma) noexcept {
    _sigma = sigma;
    return *this;
}

template <typename T>
typename MaxwellSigmaGenerator<T>::Builder&
MaxwellSigmaGenerator<T>::Builder::with_seed(const unsigned int seed) noexcept {
    _seed = seed;
    return *this;
}

template <typename T>
void
MaxwellSigmaGenerator<T>::Builder::validate() const {
    if (!_sigma.has_value()) {
        atlas::logger::error()
            << "MaxwellSigmaGenerator::Builder: sigma must be provided.";
        throw std::runtime_error("MaxwellSigmaGenerator::Builder: sigma must be provided.");
    }

    if (!(*_sigma > T(0))) {
        atlas::logger::error()
            << "MaxwellSigmaGenerator::Builder: sigma must be greater than zero.";
        throw std::runtime_error("MaxwellSigmaGenerator::Builder: sigma must be greater than zero.");
    }
}

}