#pragma once
#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>
namespace atlas {
namespace solver {
    enum class DsmcMode : int {
        Basic,
        Advanced
    };
    template <typename T>
    struct DsmcDeviceProbe {
        T* g_ref_per_cell { nullptr };
        int* n_collisions { nullptr };
        int* collision_offset { nullptr };
    };
    template <typename T>
    class DsmcData {
    public:
        DsmcData();
        ~DsmcData() = default;
        ATLAS_HOST ATLAS_FORCE_INLINE DsmcDeviceProbe<T>
        make_device_probe() noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        reset(int n_cells);
        ATLAS_HOST ATLAS_FORCE_INLINE T
        sigma_t() const;
        ATLAS_HOST ATLAS_FORCE_INLINE T
        cell_volume() const;
        ATLAS_HOST ATLAS_FORCE_INLINE T
        number_weight() const;
        ATLAS_HOST ATLAS_FORCE_INLINE int
        total_collisions() const;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_total_collisions(int v);

    private:
        DeviceBuffer<T> d_g_ref_per_cell;
        DeviceBuffer<int> d_n_collisions;
        DeviceBuffer<int> d_collision_offset;
        T _number_weight      = T(1);
        T _d                  = T(1);
        T _sigma_t            = T(1);
        T _cell_volume        = T(1);
        int _total_collisions = 0;
    };
}
template <typename T>
using DsmcData = solver::DsmcData<T>;
template <typename T>
using DsmcDataHostPtr = atlas::host_shared_ptr<solver::DsmcData<T>>;
template <typename T>
using DsmcDataDevicePtr = atlas::device_shared_ptr<solver::DsmcData<T>>;
}
#include <atlas/solver/dsmc/dsmc_data.hpp>