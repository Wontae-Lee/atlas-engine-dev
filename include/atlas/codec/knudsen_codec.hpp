#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename KnudsenCodec<T>::Builder
KnudsenCodec<T>::builder() noexcept {
    // Create and return a new builder object for staged
    // `KnudsenCodec<T>` construction.
    //
    // This is the preferred entry point when required construction
    // parameters such as the simulation domain and characteristic
    // length are supplied incrementally before final instantiation.
    return Builder {};
}

template <typename T>
KnudsenCodec<T>::KnudsenCodec(const DomainHostPtr<T>& domain, T characteristic_length)
    : Codec<T>(domain)
    , _characteristic_length(characteristic_length) {
    // Store the characteristic reference length used by the codec.
    //
    // In Knudsen-based modeling, this value represents the physical
    // length scale against which microscopic transport behavior is compared.

    // The characteristic length must be strictly positive.
    //
    // A zero or negative value would make the Knudsen interpretation invalid
    // and would typically lead to undefined or non-physical normalization.
    atlas::check<std::invalid_argument>(characteristic_length > T(0))
        << "KnudsenCodec: characteristic_length must be positive.";

    // Query the total number of simulation cells from the domain.
    // The codec stores one Knudsen-related scalar per cell.
    const auto num_of_cells = domain->number_of_cells();

    // Allocate and initialize the device-side storage for per-cell
    // Knudsen values.
    //
    // Initial value:
    // - `T(0)` means the field starts cleared before any encode step
    //   populates it from runtime simulation state.
    d_knudsen_values.resize(num_of_cells, T(0));
}

template <typename T>
void
KnudsenCodec<T>::encode(const FluidDeviceProbe<T>& particle_probe,
                        const DomainDeviceProbe<T>&,
                        const SpatialHashingProbe<T>&,
                        CodecDeviceProbe<T>&) {
    // The particle probe is part of the intended future data path,
    // but the current stub does not use it yet.
    (void)particle_probe;

    // Placeholder for future encoding logic.
    //
    // Intended responsibilities may include:
    // - reading particle or local flow state,
    // - estimating mean-free-path-related quantities,
    // - computing a Knudsen number or equivalent per-cell metric,
    // - writing the result into `d_knudsen_values`.
}

template <typename T>
void
KnudsenCodec<T>::decode(const FluidDeviceProbe<T>&,
                        const DomainDeviceProbe<T>&,
                        const SpatialHashingProbe<T>&,
                        CodecDeviceProbe<T>&) {
    // Placeholder for future decoding logic.
    //
    // Intended responsibilities may include:
    // - consuming codec-side Knudsen data,
    // - applying derived classifications or corrections,
    // - pushing the decoded information back into runtime-visible state.
    //
    // Parameters are intentionally unnamed because the current
    // implementation does not use them yet.
}

template <typename T>
CodecType
KnudsenCodec<T>::type() const noexcept {
    // Return the runtime codec type tag identifying this
    // implementation as the Knudsen codec variant.
    //
    // This can be used by dispatch logic, device probe metadata,
    // diagnostics, or serialization code.
    return CodecType::knudsen;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_domain(DomainHostPtr<T> domain) noexcept {
    // Store the simulation domain inside the builder.
    //
    // The builder retains this dependency until final construction,
    // allowing the caller to assemble required parameters in stages.
    _domain = std::move(domain);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_characteristic_length(T characteristic_length) noexcept {
    // Store the characteristic physical length in the builder.
    //
    // This value is preserved until `build()` or `make_host_shared()`
    // materializes the final codec instance.
    _characteristic_length = characteristic_length;
    return *this;
}

template <typename T>
void
KnudsenCodec<T>::Builder::validate() const {
    // Ensure that a valid simulation domain has been provided.
    //
    // The codec depends on the domain for sizing internal storage
    // and for interpreting per-cell encoded data.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "KnudsenCodec::Builder: domain must not be null.";

    // Ensure that the physical reference length is valid.
    //
    // Knudsen-based calculations require a strictly positive
    // characteristic scale.
    atlas::check<std::invalid_argument>(_characteristic_length > T(0))
        << "KnudsenCodec::Builder: characteristic_length must be positive.";
}

template <typename T>
KnudsenCodec<T>
KnudsenCodec<T>::Builder::build() const {
    // Validate all staged builder parameters before constructing
    // the final codec object.
    validate();

    // Construct and return the codec by value using the validated
    // domain and characteristic length.
    return KnudsenCodec<T>(_domain, _characteristic_length);
}

template <typename T>
atlas::host_shared_ptr<KnudsenCodec<T>>
KnudsenCodec<T>::Builder::make_host_shared() const {
    // Validate all staged builder parameters before allocating
    // the codec in shared host-managed storage.
    validate();

    // Construct the codec directly in host-shared memory and
    // return the resulting shared pointer.
    return atlas::make_host_shared<KnudsenCodec<T>>(_domain, _characteristic_length);
}

} // namespace atlas::system