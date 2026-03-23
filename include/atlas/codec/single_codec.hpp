#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename SingleCodec<T>::Builder
SingleCodec<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
SingleCodec<T>::SingleCodec(const DomainHostPtr<T>& domain)
    : Codec<T>(domain) {

    this->reset();
}

template <typename T>
void
SingleCodec<T>::encode(const ParticleDeviceProbe<T>&,
                       const DomainDeviceProbe<T>&,
                       const SpatialHashingProbe<T>&,
                       CodecDeviceProbe<T>&) {
}

template <typename T>
void
SingleCodec<T>::decode(const ParticleDeviceProbe<T>&,
                       const DomainDeviceProbe<T>&,
                       const SpatialHashingProbe<T>&,
                       CodecDeviceProbe<T>&) {
}

template <typename T>
CodecType
SingleCodec<T>::type() const noexcept {

    return CodecType::single;
}

template <typename T>
typename SingleCodec<T>::Builder&
SingleCodec<T>::Builder::with_domain(DomainHostPtr<T> domain) noexcept {

    _domain = std::move(domain);
    return *this;
}

template <typename T>
void
SingleCodec<T>::Builder::validate() const {

    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "SingleCodec::Builder: domain must not be null.";
}

template <typename T>
SingleCodec<T>
SingleCodec<T>::Builder::build() const {

    validate();
    return SingleCodec<T>(_domain);
}

template <typename T>
atlas::host_shared_ptr<SingleCodec<T>>
SingleCodec<T>::Builder::make_host_shared() const {

    validate();
    return atlas::make_host_shared<SingleCodec<T>>(_domain);
}

}