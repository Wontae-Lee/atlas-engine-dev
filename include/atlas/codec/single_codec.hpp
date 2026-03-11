#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

// ============================================================
// SingleCodec
// ============================================================

template <typename T>
typename SingleCodec<T>::Builder
SingleCodec<T>::builder() noexcept {
    // Factory entry point for the fluent Builder API.
    //
    // Returning a value (not a reference) keeps usage simple and avoids
    // lifetime issues:
    //   auto codec = SingleCodec<float>::builder()
    //                 .with_domain(domain)
    //                 .make_host_shared();
    return Builder {};
}

template <typename T>
SingleCodec<T>::SingleCodec(const DomainHostPtr<T>& domain)
    : Codec<T>(domain) {
    // Forward the domain to the base Codec implementation.
    //
    // Codec<T> is responsible for:
    // - validating the domain pointer (must not be null),
    // - allocating / sizing device buffers based on the domain grid,
    // - resetting internal state to a known default.
    //
    // We call reset() again here to guarantee SingleCodec starts from a clean
    // state even if the base class behavior changes in the future.
    //
    // (If this becomes redundant, it is still harmless: reset() is idempotent
    // with respect to "clear to defaults".)
    this->reset();
}

template <typename T>
void
SingleCodec<T>::encode(const ParticleDeviceProbe<T>& /*particle_probe*/,
                       const DomainDeviceProbe<T>& /*domain_probe*/,
                       const SpatialHashingProbe<T>& /*searcher_probe*/,
                       CodecDeviceProbe<T>& /*codec_probe*/) {
    // Encode step for this codec.
    //
    // SingleCodec is intentionally a pass-through / identity codec:
    // - It does not compress, quantize, or remap particle data.
    // - It does not allocate per-cell solver resources.
    // - It does not write any state into the codec buffers.
    //
    // Keeping this as an explicit no-op is useful because it satisfies the
    // common Codec interface, allowing the system to swap codecs without
    // adding special-case branching.
}

template <typename T>
void
SingleCodec<T>::decode(const ParticleDeviceProbe<T>& /*particle_probe*/,
                       const DomainDeviceProbe<T>& /*domain_probe*/,
                       const SpatialHashingProbe<T>& /*searcher_probe*/,
                       CodecDeviceProbe<T>& /*codec_probe*/) {
    // Decode step for this codec.
    //
    // Symmetric with encode(): SingleCodec does not transform any state, so
    // there is nothing to reconstruct or apply.
    //
    // This function exists to preserve the encode/decode pipeline structure.
}

template <typename T>
CodecType
SingleCodec<T>::type() const noexcept {
    // Identify this codec as a "single" codec.
    return CodecType::single;
}

// ============================================================
// Builder
// ============================================================

template <typename T>
typename SingleCodec<T>::Builder&
SingleCodec<T>::Builder::with_domain(DomainHostPtr<T> domain) noexcept {
    // Store the domain that will be used to construct the codec.
    //
    // Move semantics:
    // - DomainHostPtr<T> is typically a shared-pointer-like handle.
    // - Moving avoids an unnecessary refcount bump (when applicable) and
    //   makes intent explicit: the builder owns the configuration.
    _domain = std::move(domain);
    return *this;
}

template <typename T>
void
SingleCodec<T>::Builder::validate() const {
    // Ensure the builder was configured with a valid domain.
    //
    // A codec depends on the domain to size per-cell storage and to provide
    // consistent world->grid mapping. Constructing without a domain would
    // lead to undefined behavior later (wrong buffer sizes, invalid indexing).
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "SingleCodec::Builder: domain must not be null.";
}

template <typename T>
SingleCodec<T>
SingleCodec<T>::Builder::build() const {
    // Construct a codec instance by value.
    //
    // This is convenient for stack allocation or embedding, but note that
    // the codec owns device buffers, so copying may be expensive or disabled.
    // Returning by value assumes the type is movable (typical for resource owners).
    validate();
    return SingleCodec<T>(_domain);
}

template <typename T>
atlas::host_shared_ptr<SingleCodec<T>>
SingleCodec<T>::Builder::make_host_shared() const {
    // Construct a codec managed by a host-side shared pointer.
    //
    // This is often the preferred ownership model in the engine because:
    // - codecs can be shared between systems,
    // - lifetime is explicit and safe across asynchronous compute pipelines,
    // - it matches the style of other engine components (domains, searchers, etc.).
    validate();
    return atlas::make_host_shared<SingleCodec<T>>(_domain);
}

} // namespace atlas::system
