#ifndef ATLAS_ENGINE_DEV_SOLVER_HPP
#define ATLAS_ENGINE_DEV_SOLVER_HPP

#include <atlas/searcher/spatial_hashing_searcher.h>

namespace atlas::solver {

template <typename T>
Solver<T>::Solver() {
    _searcher = atlas::make_host_shared<SpatialHashingSearcher<T>>();
}

template <typename T>
void
Solver<T>::operator()(const system::ParticleDeviceProbe<T>& data,
                      T dt,
                      int& active) {
    _searcher->build(data, active);
    solve(data, dt, active);
}

}

#endif