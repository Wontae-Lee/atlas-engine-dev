#pragma once
#include <atlas/logging/logging.h>
namespace atlas::system {
template <typename T>
typename DeepLearningCodec<T>::Builder
DeepLearningCodec<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
DeepLearningCodec<T>::DeepLearningCodec(UniverseHostPtr<T> domain,
                                        FluidHostPtr<T> fluid,
                                        SpatialHashingSearcherHostPtr<T> searcher)
    : Codec<T>(std::move(domain), std::move(fluid), std::move(searcher)) {
    this->reset();
}

template <typename T>
void
DeepLearningCodec<T>::encode() {
    typename Codec<T>::CodecProbe probe;
    static_cast<void>(this->make_probe(probe));
}

template <typename T>
void
DeepLearningCodec<T>::decode() {
    typename Codec<T>::CodecProbe probe;
    static_cast<void>(this->make_probe(probe));
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_domain(UniverseHostPtr<T> domain) noexcept {
    _domain = std::move(domain);
    return *this;
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
void
DeepLearningCodec<T>::Builder::validate() const {
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
    validate();
    return DeepLearningCodec<T>(_domain, _fluid, _searcher);
}

template <typename T>
atlas::host_shared_ptr<DeepLearningCodec<T>>
DeepLearningCodec<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<DeepLearningCodec<T>>(_domain, _fluid, _searcher);
}

}
