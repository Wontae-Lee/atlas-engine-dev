#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename KnudsenCodec<T>::Builder
KnudsenCodec<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
KnudsenCodec<T>::KnudsenCodec(const DomainHostPtr<T>& domain, T characteristic_length)
    : Codec<T>(domain)
    , _characteristic_length(characteristic_length) {

    // Knudsen evaluation is only meaningful for a positive reference length.
    atlas::check<std::invalid_argument>(characteristic_length > T(0))
        << "KnudsenCodec: characteristic_length must be positive.";

    const auto num_of_cells = domain->number_of_cells();
    // Store one Knudsen value per domain cell.
    d_knudsen_values.resize(num_of_cells, T(0));
}

template <typename T>
void
KnudsenCodec<T>::encode(const FluidDeviceProbe<T>& particle_probe,
                        const DomainDeviceProbe<T>&,
                        const SpatialHashingProbe<T>&,
                        CodecDeviceProbe<T>&) {

    (void)particle_probe;
    // Placeholder: populate d_knudsen_values from the current simulation state.
}

template <typename T>
void
KnudsenCodec<T>::decode(const FluidDeviceProbe<T>&,
                        const DomainDeviceProbe<T>&,
                        const SpatialHashingProbe<T>&,
                        CodecDeviceProbe<T>&) {
    // Placeholder: apply codec-side Knudsen data back into runtime state.
}

template <typename T>
CodecType
KnudsenCodec<T>::type() const noexcept {

    return CodecType::knudsen;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_domain(DomainHostPtr<T> domain) noexcept {

    // Builders take ownership of dependencies until construction.
    _domain = std::move(domain);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_characteristic_length(T characteristic_length) noexcept {

    // Persist the physical scale until build() materializes the codec.
    _characteristic_length = characteristic_length;
    return *this;
}

template <typename T>
void
KnudsenCodec<T>::Builder::validate() const {

    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "KnudsenCodec::Builder: domain must not be null.";

    atlas::check<std::invalid_argument>(_characteristic_length > T(0))
        << "KnudsenCodec::Builder: characteristic_length must be positive.";
}

template <typename T>
KnudsenCodec<T>
KnudsenCodec<T>::Builder::build() const {

    validate();
    return KnudsenCodec<T>(_domain, _characteristic_length);
}

template <typename T>
atlas::host_shared_ptr<KnudsenCodec<T>>
KnudsenCodec<T>::Builder::make_host_shared() const {

    validate();
    return atlas::make_host_shared<KnudsenCodec<T>>(_domain, _characteristic_length);
}

}
