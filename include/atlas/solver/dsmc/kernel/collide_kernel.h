#pragma once

#include <atlas/math/math.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/solver/dsmc/dsmc_data.h>
#include <atlas/system/particle_data.h>

namespace atlas::solver::dsmc {
namespace kernel {
    template <typename T>
    class CollideKernel {
    public:
        CollideKernel()          = default;
        virtual ~CollideKernel() = default;

        ATLAS_HOST ATLAS_FORCE_INLINE virtual void
        operator()(const system::ParticleDeviceProbe<T>& data,
                   const DsmcDeviceProbe<T>& dsmc_probe,
                   const system::SpatialHashProbe<T>& neighbor_probe,
                   DeviceBuffer<int>& d_flattened_collision) const = 0;

        ATLAS_HOST ATLAS_FORCE_INLINE virtual void
        collide(const system::ParticleDeviceProbe<T>& data,
                const DsmcDeviceProbe<T>& dsmc_probe,
                const system::SpatialHashProbe<T>& neighbor_probe,
                DeviceBuffer<int>& d_flattened_collision) const = 0;
    };
}

template <typename T>
using CollideKernel = kernel::CollideKernel<T>;
template <typename T>
using CollideKernelHostPtr = atlas::host_shared_ptr<kernel::CollideKernel<T>>;
template <typename T>
using CollideKernelDevicePtr = atlas::device_shared_ptr<kernel::CollideKernel<T>>;

}