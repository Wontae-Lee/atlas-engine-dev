#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename KnudsenCodec<T>::Builder
KnudsenCodec<T>::builder() noexcept {
    // Builder entry point (fluent configuration API).
    //
    // Typical usage:
    //   auto codec = KnudsenCodec<float>::builder()
    //                 .with_domain(domain)
    //                 .with_characteristic_length(L)
    //                 .make_host_shared();
    //
    // Returning a fresh Builder by value keeps ownership simple and avoids
    // dangling references.
    return Builder {};
}

template <typename T>
KnudsenCodec<T>::KnudsenCodec(const DomainHostPtr<T>& domain, T characteristic_length)
    : Codec<T>(domain) // Base class validates the domain and sizes base buffers.
    , _characteristic_length(characteristic_length) {

    // Characteristic length (L) is a physical scale used in Knudsen number:
    //   Kn = lambda / L
    // Therefore L must be strictly positive to avoid division by zero and
    // to keep the quantity physically meaningful.
    atlas::check<std::invalid_argument>(characteristic_length > T(0))
        << "KnudsenCodec: characteristic_length must be positive.";

    // Allocate per-cell storage for the computed Knudsen metric.
    //
    // The codec computes a scalar value per grid cell (e.g., local Knudsen number
    // or a proxy for regime classification). The domain determines how many cells
    // exist, so the buffer must match `domain->number_of_cells()`.
    //
    // Initializing to 0 provides a safe default before the first encode().
    const auto num_of_cells = domain->number_of_cells();
    d_knudsen_values.resize(num_of_cells, T(0));
}

template <typename T>
void
KnudsenCodec<T>::encode(const ParticleDeviceProbe<T>& particle_probe,
                        const DomainDeviceProbe<T>& /*domain_probe*/,
                        const SpatialHashingProbe<T>& /*searcher_probe*/,
                        CodecDeviceProbe<T>& /*codec_probe*/) {

    // Encode phase: compute and/or store information needed for later steps.
    //
    // In a Knudsen-based codec, the typical objective is:
    // - compute a per-cell (or per-particle) Knudsen number, or a regime flag
    //   derived from it (continuum / transitional / free molecular),
    // - optionally mark which solver model to use per cell (e.g., CFD vs DSMC),
    // - store the result into a device buffer for consumption by later kernels.
    //
    // Common high-level algorithm outline (not implemented here):
    //  1) Clear/initialize per-cell accumulators (counts, densities, etc.).
    //  2) For each particle:
    //     - map position to cell id (domain world->grid mapping),
    //     - atomically accumulate per-cell quantities (number density, temperature,
    //       mean speed, etc.), depending on your model.
    //  3) For each cell:
    //     - derive mean free path (lambda) from local quantities,
    //     - compute Kn = lambda / characteristic_length (L),
    //     - write Kn (or a derived classification) into d_knudsen_values.
    //
    // Notes:
    // - `particle_probe` provides device pointers and particle_count count.
    // - `domain_probe` and `searcher_probe` could be used to accelerate
    //   neighbor-based estimates (e.g., local density from neighbors).
    // - `codec_probe` could point to per-cell solver selection flags.
    //
    // TODO: Implement Knudsen number calculation here.
    // TODO: Not implemented yet.

    (void)particle_probe; // Silence unused warnings until implemented.
}

template <typename T>
void
KnudsenCodec<T>::decode(const ParticleDeviceProbe<T>& /*particle_probe*/,
                        const DomainDeviceProbe<T>& /*domain_probe*/,
                        const SpatialHashingProbe<T>& /*searcher_probe*/,
                        CodecDeviceProbe<T>& /*codec_probe*/) {
    // Decode phase: apply stored codec state back into the simulation representation.
    //
    // Many codecs are symmetric (encode produces a representation; decode restores it).
    // For a regime-classification codec, decode may be a no-op if:
    // - the knudsen values are only consumed by solver selection logic directly, or
    // - encode already writes the required per-cell flags used later.
    //
    // If decode is required, it might:
    // - write regime flags into `codec_probe.allocated_system` (or another buffer),
    // - remap/restore per-particle properties if encode modified them (unlikely here).
    //
    // TODO: If decoding is needed, implement here.
    // TODO: Not implemented yet.
}

template <typename T>
CodecType
KnudsenCodec<T>::type() const noexcept {
    // Identify this codec's type.
    return CodecType::knudsen;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_domain(DomainHostPtr<T> domain) noexcept {
    // Configure the domain dependency.
    //
    // The domain defines:
    // - grid resolution,
    // - number of cells (buffer sizing),
    // - mapping between world coordinates and cell indices.
    //
    // Move semantics avoid extra refcount churn for shared-pointer-like handles.
    _domain = std::move(domain);
    return *this;
}

template <typename T>
typename KnudsenCodec<T>::Builder&
KnudsenCodec<T>::Builder::with_characteristic_length(T characteristic_length) noexcept {
    // Configure the characteristic length (L) used in Knudsen number evaluation.
    //
    // This value is stored and validated at build time. Keeping it in the builder
    // makes it explicit and prevents partially-constructed codecs.
    _characteristic_length = characteristic_length;
    return *this;
}

template <typename T>
void
KnudsenCodec<T>::Builder::validate() const {
    // Validate that all mandatory configuration has been provided and is sensible.
    //
    // 1) Domain must exist (buffer sizing and mapping depend on it).
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "KnudsenCodec::Builder: domain must not be null.";

    // 2) Characteristic length must be positive (Kn = lambda / L).
    atlas::check<std::invalid_argument>(_characteristic_length > T(0))
        << "KnudsenCodec::Builder: characteristic_length must be positive.";
}

template <typename T>
KnudsenCodec<T>
KnudsenCodec<T>::Builder::build() const {
    // Build a codec instance by value.
    //
    // This is useful for stack allocation or embedding in higher-level objects.
    // The codec owns device buffers, so returning by value assumes the type is
    // movable and that copying is either cheap or disabled.
    validate();
    return KnudsenCodec<T>(_domain, _characteristic_length);
}

template <typename T>
atlas::host_shared_ptr<KnudsenCodec<T>>
KnudsenCodec<T>::Builder::make_host_shared() const {
    // Build a codec managed by host_shared_ptr.
    //
    // Preferred when the codec is shared across systems or stored in polymorphic
    // containers, and when lifetime must be managed across asynchronous work.
    validate();
    return atlas::make_host_shared<KnudsenCodec<T>>(_domain, _characteristic_length);
}

} // namespace atlas::system
