#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
typename DeepLearningCodec<T>::Builder
DeepLearningCodec<T>::builder() noexcept {

    // Return a default-initialized builder for fluent DeepLearningCodec construction.
    return Builder {};
}

template <typename T>
DeepLearningCodec<T>::DeepLearningCodec(UniverseHostPtr<T> domain,
                                        FluidHostPtr<T> fluid,
                                        SpatialHashingSearcherHostPtr<T> searcher)
    : Codec<T>(std::move(domain), std::move(fluid), std::move(searcher)) {

    // Initialize or restore the codec state after the base dependencies
    // have been installed.
    this->reset();
}

template <typename T>
void
DeepLearningCodec<T>::encode() {
    // Encoding logic is not implemented yet.
    //
    // This function currently acts as an extension point for future
    // deep-learning-based feature extraction or latent-state encoding.
}

template <typename T>
void
DeepLearningCodec<T>::decode() {
    // Decoding logic is not implemented yet.
    //
    // This function currently acts as an extension point for future
    // reconstruction or inference-based state decoding.
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_domain(UniverseHostPtr<T> domain) noexcept {

    // Store the universe/domain dependency for later construction.
    _domain = std::move(domain);
    return *this;
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {

    // Store the fluid dependency for later construction.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {

    // Store the spatial hashing searcher dependency for later construction.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
void
DeepLearningCodec<T>::Builder::validate() const {

    // All required dependencies must be present before a valid codec can be built.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "DeepLearningCodec::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "DeepLearningCodec::Builder: fluid must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "DeepLearningCodec::Builder: searcher must not be null.";
}

template <typename T>
DeepLearningCodec<T>
DeepLearningCodec<T>::Builder::build() const {

    // Validate the builder configuration before constructing a value object.
    validate();

    return DeepLearningCodec<T>(_domain, _fluid, _searcher);
}

template <typename T>
atlas::host_shared_ptr<DeepLearningCodec<T>>
DeepLearningCodec<T>::Builder::make_host_shared() const {

    // Validate the builder configuration before constructing a shared instance.
    validate();

    return atlas::make_host_shared<DeepLearningCodec<T>>(_domain, _fluid, _searcher);
}

} // namespace atlas::system