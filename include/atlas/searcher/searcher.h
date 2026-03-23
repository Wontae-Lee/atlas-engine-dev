#pragma once
#include <atlas/data/particle_data.h>
#include <atlas/memory/memory.h>

namespace atlas::system {

enum class NeighborSearchRange : int {
    single,
    multiple
};

template <typename T>
class Searcher {
public:
    Searcher() = default;

    virtual ~Searcher() = default;

    virtual void
    build(const system::ParticleDeviceProbe<T>& particle_probe)
        = 0;
};

}

namespace atlas {

template <typename T>
using Searcher = system::Searcher<T>;

template <typename T>
using SearcherHostPtr = atlas::host_shared_ptr<system::Searcher<T>>;

template <typename T>
using SearcherDevicePtr = atlas::device_shared_ptr<system::Searcher<T>>;

}