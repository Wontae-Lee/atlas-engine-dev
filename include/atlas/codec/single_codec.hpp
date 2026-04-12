#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename SingleCodec<T>::Builder
SingleCodec<T>::builder() noexcept {
    // Create and return a fresh builder object for staged
    // `SingleCodec<T>` construction.
    //
    // This builder-based entry point is useful when required
    // dependencies such as the simulation domain are supplied
    // incrementally before the final codec instance is created.
    return Builder {};
}

template <typename T>
SingleCodec<T>::SingleCodec(const DomainHostPtr<T>& domain)
    : Codec<T>(domain) {
    // The base `Codec<T>` constructor already:
    // - stores the domain handle,
    // - validates that the domain is not null,
    // - initializes shared codec-side state.
    //
    // Call `reset()` explicitly here so this concrete codec begins
    // from a clean and fully initialized state under its own type.
    this->reset();
}

template <typename T>
void
SingleCodec<T>::encode(const FluidDeviceProbe<T>&,
                       const DomainDeviceProbe<T>&,
                       const SpatialHashingProbe<T>&,
                       CodecDeviceProbe<T>&) {
    // The single-state codec does not transform the simulation into
    // an alternate encoded representation.
    //
    // Intended semantics:
    // - the runtime state is already kept in its direct form,
    // - no additional compression, abstraction, or latent projection
    //   is performed during the encode step.
    //
    // Parameters are intentionally unnamed because the current
    // implementation does not need to consume them.
}

template <typename T>
void
SingleCodec<T>::decode(const FluidDeviceProbe<T>&,
                       const DomainDeviceProbe<T>&,
                       const SpatialHashingProbe<T>&,
                       CodecDeviceProbe<T>&) {
    // No decode/reconstruction phase is required for `SingleCodec`.
    //
    // Since the simulation state is preserved in its direct single
    // representation, there is nothing to reconstruct from an
    // intermediate or encoded form.
    //
    // Parameters are intentionally unnamed because the current
    // implementation does not need to consume them.
}

template <typename T>
CodecType
SingleCodec<T>::type() const noexcept {
    // Return the runtime codec type tag identifying this
    // implementation as the single-state codec variant.
    //
    // This value can be used by higher-level dispatch logic,
    // diagnostics, serialization, or device-side metadata.
    return CodecType::single;
}

template <typename T>
typename SingleCodec<T>::Builder&
SingleCodec<T>::Builder::with_domain(DomainHostPtr<T> domain) noexcept {
    // Store the simulation domain inside the builder's staged state.
    //
    // The builder retains this dependency until final construction
    // through `build()` or `make_host_shared()`.
    //
    // `std::move` is used so the domain handle follows the ownership
    // and transfer semantics of `DomainHostPtr<T>`.
    _domain = std::move(domain);
    return *this;
}

template <typename T>
void
SingleCodec<T>::Builder::validate() const {
    // Ensure that the builder has been provided with a valid domain
    // before attempting to construct the codec.
    //
    // A `SingleCodec` still depends on the simulation domain because
    // its base codec infrastructure uses the domain to size and
    // initialize runtime-managed state.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "SingleCodec::Builder: domain must not be null.";
}

template <typename T>
SingleCodec<T>
SingleCodec<T>::Builder::build() const {
    // Validate the staged builder configuration before creating
    // the final codec object.
    validate();

    // Construct and return the codec by value using the validated domain.
    return SingleCodec<T>(_domain);
}

template <typename T>
atlas::host_shared_ptr<SingleCodec<T>>
SingleCodec<T>::Builder::make_host_shared() const {
    // Validate the staged builder configuration before allocating
    // the codec in shared host-managed storage.
    validate();

    // Construct the codec directly in host-shared memory and return
    // the resulting shared pointer.
    return atlas::make_host_shared<SingleCodec<T>>(_domain);
}

} // namespace atlas::system