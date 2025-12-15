#pragma once
#include <atlas/buffer/device_buffer.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

namespace atlas {
namespace system {
    template <typename T>
    struct ParticleDeviceProbe {
        Vector3<T>* pos { nullptr };
        Vector3<T>* vel { nullptr };
        size_t* species { nullptr };
        int alive { 0 };
    };

    template <typename T>
    class ParticleData {
    public:
        ATLAS_HOST ATLAS_FORCE_INLINE explicit ParticleData(size_t buffer_size);
        ~ParticleData() = default;
        ATLAS_HOST ATLAS_FORCE_INLINE ParticleDeviceProbe<T>
        make_device_probe() noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
        positions() noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
        velocities() noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<size_t>&
        species() noexcept;

    private:
        DeviceBuffer<Vector3<T>> d_pos;
        DeviceBuffer<Vector3<T>> d_vel;
        DeviceBuffer<size_t> d_species;
    };
}

template <typename T>
using ParticleData = system::ParticleData<T>;
template <typename T>
using ParticleDataHostPtr = atlas::host_shared_ptr<system::ParticleData<T>>;
}

#include <atlas/system/particle_data.hpp>