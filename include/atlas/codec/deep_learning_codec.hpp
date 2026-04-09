#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename DeepLearningCodec<T>::Builder
DeepLearningCodec<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
DeepLearningCodec<T>::DeepLearningCodec(const DomainHostPtr<T>& domain)
    : Codec<T>(domain) {

    this->reset();
}

template <typename T>
void
DeepLearningCodec<T>::encode(const FluidDeviceProbe<T>&,
                             const DomainDeviceProbe<T>&,
                             const SpatialHashingProbe<T>&,
                             CodecDeviceProbe<T>&) {
}

template <typename T>
void
DeepLearningCodec<T>::decode(const FluidDeviceProbe<T>&,
                             const DomainDeviceProbe<T>&,
                             const SpatialHashingProbe<T>&,
                             CodecDeviceProbe<T>&) {
}

template <typename T>
CodecType
DeepLearningCodec<T>::type() const noexcept {

    return CodecType::deep_learning;
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_domain(DomainHostPtr<T> domain) noexcept {

    _domain = std::move(domain);
    return *this;
}

template <typename T>
void
DeepLearningCodec<T>::Builder::validate() const {

    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "DeepLearningCodec::Builder: domain must not be null.";
}

template <typename T>
DeepLearningCodec<T>
DeepLearningCodec<T>::Builder::build() const {

    validate();
    return DeepLearningCodec<T>(_domain);
}

template <typename T>
atlas::host_shared_ptr<DeepLearningCodec<T>>
DeepLearningCodec<T>::Builder::make_host_shared() const {

    validate();
    return atlas::make_host_shared<DeepLearningCodec<T>>(_domain);
}

}
