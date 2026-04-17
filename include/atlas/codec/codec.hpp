#pragma once

namespace atlas::system {

template <typename T>
Codec<T>::Codec(UniverseHostPtr<T> domain,
                FluidHostPtr<T> fluid,
                SpatialHashingSearcherHostPtr<T> searcher)
    : _domain(std::move(domain))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher)) {

    atlas::check<std::invalid_argument>(static_cast<bool>(_domain))
        << "Codec: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "Codec: fluid must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "Codec: searcher must not be null.";

    reset();
}

template <typename T>
void
Codec<T>::update() {

    this->encode();
    this->decode();
}

template <typename T>
void
Codec<T>::reset() noexcept {

    const auto num_of_cells = _domain->number_of_cells();

    d_allocated_solver.resize(num_of_cells, 0);
}

template <typename T>
DeviceBuffer<int>&
Codec<T>::allocated_solver() noexcept {

    return d_allocated_solver;
}

template <typename T>
const DeviceBuffer<int>&
Codec<T>::allocated_solver() const noexcept {

    return d_allocated_solver;
}

}
