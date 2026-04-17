#pragma once

namespace atlas::system {

template <typename T>
Measure<T>::Measure(UniverseHostPtr<T> universe,
                    FluidHostPtr<T> fluid,
                    SpatialHashingSearcherHostPtr<T> searcher) noexcept
    : _universe(std::move(universe))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher)) {
}

}
