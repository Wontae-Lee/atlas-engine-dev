#ifndef INCLUDE_ATLAS_SYSTEM_PARTICLE_DATA_H
#define INCLUDE_ATLAS_SYSTEM_PARTICLE_DATA_H

#include <atlas/buffer/device_buffer.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

namespace atlas {
namespace system {
    template <typename T>
    struct ParticleDeviceProbe {
        Vector3<T>* pos { nullptr };
        Vector3<T>* vel { nullptr };
    };

    template <typename T>
    class ParticleData {
    public:
        ATLAS_HOST ATLAS_FORCE_INLINE explicit ParticleData(int capacity);

        ATLAS_HOST ATLAS_FORCE_INLINE ParticleDeviceProbe<T>
        make_device_probe() noexcept;

        ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
        positions() noexcept;

        ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
        velocities() noexcept;

    private:
        DeviceBuffer<Vector3<T>> d_pos;
        DeviceBuffer<Vector3<T>> d_vel;
    };

}

template <typename T>
using ParticleData = system::ParticleData<T>;

template <typename T>
using ParticleDataHostPtr = atlas::host_shared_ptr<system::ParticleData<T>>;

}
#include <atlas/system/particle_data.hpp>

#endif