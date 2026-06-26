#pragma once

#include <atlas/logging/logging.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
typename DeepLearningCodec<T>::Builder
DeepLearningCodec<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
DeepLearningCodec<T>::DeepLearningCodec(UniverseHostPtr<T> domain,
                                        FluidHostPtr<T> fluid,
                                        SpatialHashingSearcherHostPtr<T> searcher)
    : Codec<T>(std::move(domain), std::move(fluid), std::move(searcher)) { }

template <typename T>
void
DeepLearningCodec<T>::encode() {
    static_cast<void>(this->make_probe());
}

template <typename T>
void
DeepLearningCodec<T>::decode() {
    static_cast<void>(this->make_probe());
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
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_fixed_solver(DeviceBuffer<int> fixed_solver) noexcept {
    _fixed_solver = std::move(fixed_solver);
    return *this;
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_fixed_region(DeviceBuffer<int> fixed_region) noexcept {
    _fixed_region = std::move(fixed_region);
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

    const auto cell_count = static_cast<std::size_t>(_domain->number_of_cells());
    atlas::check<std::invalid_argument>(_fixed_solver.empty() || _fixed_solver.size() == cell_count)
        << "DeepLearningCodec::Builder: fixed_solver size must match universe cell count.";
    atlas::check<std::invalid_argument>(_fixed_region.empty() || _fixed_region.size() == cell_count)
        << "DeepLearningCodec::Builder: fixed_region size must match universe cell count.";
}

template <typename T>
DeepLearningCodec<T>
DeepLearningCodec<T>::Builder::build() const {
    validate();

    auto codec = DeepLearningCodec<T>(_domain, _fluid, _searcher);
    if (!_fixed_solver.empty()) {
        codec.set_fixed_solver(_fixed_solver);
    }
    if (!_fixed_region.empty()) {
        codec.set_fixed_region(_fixed_region);
    }

    return codec;
}

template <typename T>
atlas::host_shared_ptr<DeepLearningCodec<T>>
DeepLearningCodec<T>::Builder::make_host_shared() const {
    validate();

    auto codec = atlas::make_host_shared<DeepLearningCodec<T>>(_domain, _fluid, _searcher);
    if (!_fixed_solver.empty()) {
        codec->set_fixed_solver(_fixed_solver);
    }
    if (!_fixed_region.empty()) {
        codec->set_fixed_region(_fixed_region);
    }

    return codec;
}

}