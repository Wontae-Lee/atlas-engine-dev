#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename DeepLearningCodec<T>::Builder
DeepLearningCodec<T>::builder() noexcept {
    // Builder entry point for configuring and constructing a DeepLearningCodec.
    //
    // This follows the same fluent style used across the system:
    //   auto codec = DeepLearningCodec<float>::builder()
    //                 .with_domain(domain)
    //                 .make_host_shared();
    //
    // Returning a new Builder by value keeps the API simple and avoids lifetime
    // issues that can arise from returning references to temporaries.
    return Builder {};
}

template <typename T>
DeepLearningCodec<T>::DeepLearningCodec(const DomainHostPtr<T>& domain)
    : Codec<T>(domain) {
    // The base Codec constructor:
    // - validates that `domain` is non-null,
    // - stores it for later use,
    // - and may initialize domain-dependent buffers.
    //
    // We explicitly reset to ensure this derived codec starts from a clean,
    // known state even if the base class behavior changes (defensive init).
    // This is especially useful while the class is still a stub and does not
    // yet own additional model/runtime buffers of its own.
    this->reset();
}

template <typename T>
void
DeepLearningCodec<T>::encode(const ParticleDeviceProbe<T>& /*particle_probe*/,
                             const DomainDeviceProbe<T>& /*domain_probe*/,
                             const SpatialHashingProbe<T>& /*searcher_probe*/,
                             CodecDeviceProbe<T>& /*codec_probe*/) {
    // Encode phase: transform the current simulation state into a compact or
    // model-friendly representation.
    //
    // For a deep-learning-based codec, typical responsibilities might include:
    //  - Feature extraction:
    //      * Convert particle and/or grid state into tensors (SoA -> packed features).
    //      * Examples: density, velocity moments, temperature proxies, occupancy,
    //        signed distance to boundaries, etc.
    //  - Normalization / scaling:
    //      * Apply precomputed mean/std, min/max, or log transforms to stabilize inference.
    //  - Spatial aggregation:
    //      * Accumulate particle features into per-cell descriptors (histograms, moments).
    //      * Use the spatial hash probe to estimate local neighborhoods efficiently.
    //  - Inference integration (optional, depending on architecture):
    //      * Dispatch model inference (e.g., via an external runtime) to produce
    //        per-cell or per-particle outputs.
    //  - Writeback:
    //      * Store results into device buffers used by later stages (e.g., solver selection,
    //        correction fields, learned closures).
    //
    // The provided probes indicate what data is available:
    //  - particle_probe: device pointers to particle arrays + particle_count.
    //  - domain_probe: device-friendly domain constants and grid mapping.
    //  - searcher_probe: neighbor iteration support for local feature computation.
    //  - codec_probe: a place to store codec outputs that the system consumes later.
    //
    // TODO: Not implemented yet.
}

template <typename T>
void
DeepLearningCodec<T>::decode(const ParticleDeviceProbe<T>& /*particle_probe*/,
                             const DomainDeviceProbe<T>& /*domain_probe*/,
                             const SpatialHashingProbe<T>& /*searcher_probe*/,
                             CodecDeviceProbe<T>& /*codec_probe*/) {
    // Decode phase: apply the codec output back to the simulation state.
    //
    // In a deep-learning codec, decode might:
    //  - Apply learned corrections:
    //      * Update per-particle velocities/positions (careful: stability constraints).
    //      * Update per-cell fields (pressure correction, viscosity multiplier, etc.).
    //  - Choose solver modes:
    //      * Translate inference outputs into `allocated_system` flags, thresholds,
    //        or regime classifications.
    //  - Reconstruct / denormalize:
    //      * Map model outputs back to physical units or the engine's internal units.
    //
    // Whether decode is a no-op depends on where you want effects to be applied:
    //  - If encode already writes final per-cell flags, decode may do nothing.
    //  - If encode only produces intermediate tensors, decode may perform the actual
    //    writeback to simulation buffers.
    //
    // TODO: Not implemented yet.
}

template <typename T>
CodecType
DeepLearningCodec<T>::type() const noexcept {
    // Identify this codec as a deep learning codec.
    return CodecType::deep_learning;
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_domain(DomainHostPtr<T> domain) noexcept {
    // Provide the domain dependency for this codec.
    //
    // The domain is essential because it determines:
    //  - the grid resolution and number of cells (buffer sizing),
    //  - world <-> grid mapping (feature placement / aggregation),
    //  - and the region over which the codec operates.
    //
    // Move the pointer/handle to avoid unnecessary reference-count increments.
    _domain = std::move(domain);
    return *this;
}

template <typename T>
void
DeepLearningCodec<T>::Builder::validate() const {
    // Ensure the builder has all required configuration.
    //
    // The deep learning codec requires a domain because it is typically used to
    // construct grid-aligned features and allocate per-cell storage.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "DeepLearningCodec::Builder: domain must not be null.";
}

template <typename T>
DeepLearningCodec<T>
DeepLearningCodec<T>::Builder::build() const {
    // Construct a DeepLearningCodec by value after validation.
    //
    // Useful when embedding the codec directly in other objects or for tests.
    // Once model runtime handles are introduced, this path may become mainly a
    // convenience wrapper over move construction.
    validate();
    return DeepLearningCodec<T>(_domain);
}

template <typename T>
atlas::host_shared_ptr<DeepLearningCodec<T>>
DeepLearningCodec<T>::Builder::make_host_shared() const {
    // Construct a DeepLearningCodec managed by a host_shared_ptr.
    //
    // Preferred for typical engine usage where the codec is stored polymorphically
    // or shared across systems with well-defined lifetime management.
    validate();
    return atlas::make_host_shared<DeepLearningCodec<T>>(_domain);
}

} // namespace atlas::system
