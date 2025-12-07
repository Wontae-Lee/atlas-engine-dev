#ifndef INCLUDE_ATLAS_SEARCHER_SEARCHER_H
#define INCLUDE_ATLAS_SEARCHER_SEARCHER_H

#include <atlas/buffer/device_buffer.h>
#include <atlas/memory/memory.h>
#include <atlas/system/particle_data.h>

namespace atlas {
namespace system {

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

#endif