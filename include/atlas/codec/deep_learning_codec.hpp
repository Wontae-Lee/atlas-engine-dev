#pragma once

#include <atlas/logging/logging.h>

namespace atlas {

template <typename T>
typename DeepLearningCodec<T>::Builder
DeepLearningCodec<T>::builder() noexcept {
    // Return a fresh builder so callers can configure the codec through a fluent API.
    return Builder {};
}

template <typename T>
DeepLearningCodec<T>::DeepLearningCodec(UniverseHostPtr<T> domain,
                                        FluidHostPtr<T> fluid,
                                        SpatialHashingSearcherHostPtr<T> searcher)
    : Codec<T>(std::move(domain), std::move(fluid), std::move(searcher)) {
    // Ensure codec-side buffers are initialized for the current universe layout.
    this->reset();
}

template <typename T>
void
DeepLearningCodec<T>::encode() {
    // Refresh the probe so external learning-side logic can observe current
    // simulation states and solver-control buffers.
    static_cast<void>(this->make_probe());
}

template <typename T>
void
DeepLearningCodec<T>::decode() {
    // Refresh the probe before decoding solver decisions back into the codec state.
    // The current implementation only prepares access to the required data.
    static_cast<void>(this->make_probe());
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_domain(UniverseHostPtr<T> domain) noexcept {
    // Store the universe object used to define cell count, volume, and universe states.
    _domain = std::move(domain);
    return *this;
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    // Store the fluid object used to access particle count and statistical weight.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    // Store the spatial searcher used to expose cell-wise particle traversal data.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_fixed_solver(DeviceBuffer<int> fixed_solver) noexcept {
    // Store optional per-cell solver constraints that override learned allocation.
    _fixed_solver = std::move(fixed_solver);
    return *this;
}

template <typename T>
typename DeepLearningCodec<T>::Builder&
DeepLearningCodec<T>::Builder::with_fixed_region(DeviceBuffer<int> fixed_region) noexcept {
    // Store optional per-cell region constraints that limit where learning is applied.
    _fixed_region = std::move(fixed_region);
    return *this;
}

template <typename T>
void
DeepLearningCodec<T>::Builder::validate() const {
    // The codec cannot be constructed without all simulation resources.
    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "DeepLearningCodec::Builder: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "DeepLearningCodec::Builder: fluid must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "DeepLearningCodec::Builder: searcher must not be null.";

    const auto cell_count = static_cast<std::size_t>(_domain->number_of_cells());

    // Non-empty constraint buffers must provide exactly one entry per universe cell.
    atlas::check<std::invalid_argument>(_fixed_solver.empty() || _fixed_solver.size() == cell_count)
        << "DeepLearningCodec::Builder: fixed_solver size must match universe cell count.";
    atlas::check<std::invalid_argument>(_fixed_region.empty() || _fixed_region.size() == cell_count)
        << "DeepLearningCodec::Builder: fixed_region size must match universe cell count.";
}

template <typename T>
DeepLearningCodec<T>
DeepLearningCodec<T>::Builder::build() const {
    // Validate all required resources and buffer sizes before constructing the codec.
    validate();

    auto codec = DeepLearningCodec<T>(_domain, _fluid, _searcher);

    // Apply optional fixed solver constraints only when they were explicitly provided.
    if (!_fixed_solver.empty()) {
        codec.set_fixed_solver(_fixed_solver);
    }

    // Apply optional fixed region constraints only when they were explicitly provided.
    if (!_fixed_region.empty()) {
        codec.set_fixed_region(_fixed_region);
    }

    return codec;
}

template <typename T>
atlas::host_shared_ptr<DeepLearningCodec<T>>
DeepLearningCodec<T>::Builder::make_host_shared() const {
    // Validate inputs before allocating the shared codec instance.
    validate();

    auto codec = atlas::make_host_shared<DeepLearningCodec<T>>(_domain, _fluid, _searcher);

    // Copy optional fixed solver constraints into the allocated codec instance.
    if (!_fixed_solver.empty()) {
        codec->set_fixed_solver(_fixed_solver);
    }

    // Copy optional fixed region constraints into the allocated codec instance.
    if (!_fixed_region.empty()) {
        codec->set_fixed_region(_fixed_region);
    }

    return codec;
}

} // namespace atlas