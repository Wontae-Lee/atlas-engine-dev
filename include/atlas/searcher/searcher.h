#pragma once
#include <atlas/memory/memory.h>
#include <atlas/system/particle_data.h>

namespace atlas {
namespace system {
    enum class NeighborSearchRange : int {
        single,
        multiple
    };

    enum class NeighborSearchMode : int {
        active,
        passive
    };

    template <typename T>
    class Searcher {
    public:
        Searcher()          = default;
        virtual ~Searcher() = default;
        virtual void
        build(const system::ParticleDeviceProbe<T>& data, int& active)
            = 0;
    };
}

template <typename T>
using Searcher = system::Searcher<T>;
template <typename T>
using SearcherHostPtr = atlas::host_shared_ptr<system::Searcher<T>>;
template <typename T>
using SearcherDevicePtr = atlas::device_shared_ptr<system::Searcher<T>>;
}