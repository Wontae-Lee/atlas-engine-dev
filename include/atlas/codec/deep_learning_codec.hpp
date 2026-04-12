#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename DeepLearningCodec<T>::Builder
DeepLearningCodec<T>::builder() noexcept {
    // Create and return a new builder instance for staged
    // `DeepLearningCodec<T>` construction.
    //
    // This entry point is useful when the codec dependencies,
    // especially the simulation domain, are supplied progressively
    // before final object creation.
    return Builder {};
}

template <typename T>
DeepLearningCodec<T>::DeepLearningCodec(const DomainHostPtr<T>& domain)
    : Codec<T>(domain) {
    // The base `Codec<T>` constructor already:
    // - stores the domain handle,
    // - validates that the domain is not null,
    // - initializes the base codec state.
    //
    // Call `reset()` explicitly here so the derived codec starts from a
    // known clean state even though the current deep-learning path is
    // only a placeholder implementation.
    this->reset();
}

template <typename T>
void
DeepLearningCodec<T>::encode(const FluidDeviceProbe<T>&,
                             const DomainDeviceProbe<T>&,
                             const SpatialHashingProbe<T>&,
                             CodecDeviceProbe<T>&) {
    // Placeholder for future learned encoding logic.
    //
    // Intended future responsibilities may include:
    // - reading particle state from the fluid probe,
    // - reading domain metadata from the domain probe,
    // - using spatial-neighbor information from the hashing probe,
    // - producing encoded or latent codec-side state.
    //
    // Parameters are intentionally unnamed because the current stub
    // does not yet consume them.
}

template <typename T>
void
DeepLearningCodec<T>::decode(const FluidDeviceProbe<T>&,
                             const DomainDeviceProbe<T>&,
                             const SpatialHashingProbe<T>&,
                             CodecDeviceProbe<T>&) {
    // Placeholder for future learned decoding logic.
    //
    // Intended future responsibilities may include:
    // - consuming codec-managed encoded state,
    // - reconstructing or updating simulation-facing quantities,
    // - writing decoded results back through device-visible structures.
    //
    // Parameters are intentionally unnamed because the current stub
    // does not yet consume them.
}

template <typename T>
CodecType
DeepLearningCodec<T>::type() const noexcept {
    // Return the runtime codec type tag that identifies this
    // implementation as the deep-learning codec variant.
    //
    // This tag can be used by dispatch logic, diagnostics,
    // serialization, or device-side probe metadata.
    return CodecType::deep_learning;
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_domain(DomainHostPtr<T> domain) noexcept {
    // Store the provided domain handle inside the builder.
    //
    // The builder retains this dependency until final construction
    // through either `build()` or `make_host_shared()`.
    //
    // `std::move` is used so ownership/state transfer follows the
    // semantics of the domain host pointer type.
    _domain = std::move(domain);
    return *this;
}

template <typename T>
void
DeepLearningCodec<T>::Builder::validate() const {
    // Verify that the builder has received a valid simulation domain.
    //
    // A deep-learning codec cannot be constructed without a domain because
    // its base codec infrastructure depends on domain-defined sizing and
    // spatial configuration.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "DeepLearningCodec::Builder: domain must not be null.";
}

template <typename T>
DeepLearningCodec<T>
DeepLearningCodec<T>::Builder::build() const {
    // Validate the staged builder configuration before construction.
    validate();

    // Construct and return the codec by value using the validated domain.
    return DeepLearningCodec<T>(_domain);
}

template <typename T>
atlas::host_shared_ptr<DeepLearningCodec<T>>
DeepLearningCodec<T>::Builder::make_host_shared() const {
    // Validate the staged builder configuration before allocating
    // a shared host-managed codec object.
    validate();

    // Construct the codec directly in host-shared storage and return
    // the resulting shared pointer.
    return atlas::make_host_shared<DeepLearningCodec<T>>(_domain);
}

} // namespace atlas::system