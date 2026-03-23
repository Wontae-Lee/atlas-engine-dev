#pragma once

#include <atlas/data/particle_data.h>
#include <atlas/memory/memory.h>

namespace atlas::system {

template <typename T>
class System {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE explicit System(size_t buffer_size);
    ~System() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE ParticleDataHostPtr<T>
    particle_data() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE ParticleDeviceProbe<T>&
    particle_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE const ParticleDeviceProbe<T>&
    particle_probe() const noexcept;

private:
    ParticleDataHostPtr<T> _particle_data;

    ParticleDeviceProbe<T> _particle_probe {};
};

}

namespace atlas {

template <typename T>
using System = system::System<T>;

template <typename T>
using SystemHostPtr = atlas::host_shared_ptr<system::System<T>>;

}

#include <atlas/system/system.hpp>