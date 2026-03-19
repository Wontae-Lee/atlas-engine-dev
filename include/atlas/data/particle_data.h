#pragma once

#include <cstdint>

#include <atlas/buffer/device_buffer.h>
#include <atlas/logging/logging.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

namespace atlas {
namespace system {
    template <typename T>
    struct ParticleDeviceProbe {
        Vector3<T>* pos { nullptr };
        Vector3<T>* vel { nullptr };
        size_t* species { nullptr };
        int* acitve { nullptr };
        int particle_count { 0 };
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
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE size_t
        buffer_size() const noexcept;

    private:
        DeviceBuffer<Vector3<T>> d_pos;
        DeviceBuffer<Vector3<T>> d_vel;
        DeviceBuffer<size_t> d_species;
        DeviceBuffer<int> d_active;
        size_t _buffer_size = 0;
        std::uint64_t _probe_count = 0;
    };
}

template <typename T>
using ParticleData = system::ParticleData<T>;
template <typename T>
using ParticleDataHostPtr = atlas::host_shared_ptr<system::ParticleData<T>>;

} // namespace atlas

#include <atlas/data/particle_data.hpp>
