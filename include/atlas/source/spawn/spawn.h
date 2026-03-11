#pragma once
#include <atlas/data/particle_data.h>
#include <atlas/domain/domain.h>
#include <atlas/memory/memory.h>

namespace atlas::system {

template <typename T>
class Spawn {
public:
    Spawn()          = default;
    virtual ~Spawn() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    spawn(const DomainDeviceProbe<T>& domain, const ParticleDeviceProbe<T>& particles, float dt)
        = 0;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    operator()(const DomainDeviceProbe<T>& domain, const ParticleDeviceProbe<T>& particles, float dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    activate();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    deactivate();

private:
    bool active = true;
};
}

namespace atlas {
template <typename T>
using Spawn = atlas::system::Spawn<T>;

template <typename T>
using SpawnHostPtr = atlas::host_shared_ptr<atlas::system::Spawn<T>>;

template <typename T>
using SpawnDevicePtr = atlas::device_shared_ptr<atlas::system::Spawn<T>>;

}

#include <atlas/source/spawn/spawn.hpp>