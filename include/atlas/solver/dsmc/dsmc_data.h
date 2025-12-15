#pragma once
#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>

namespace atlas {
namespace solver {
    template <typename T>
    struct DsmcDeviceProbe {
        T* g_ref_per_cell{ nullptr };
        int* n_collisions{ nullptr };
        int* n_particles_per_cell_species{ nullptr };
    };

    template <typename T>
    class DsmcData {
    public:
        ATLAS_HOST ATLAS_FORCE_INLINE
        DsmcData();
        ~DsmcData() = default;

        ATLAS_HOST ATLAS_FORCE_INLINE DsmcDeviceProbe<T>
        make_device_probe() noexcept;

        ATLAS_HOST ATLAS_FORCE_INLINE void
        reset(int n_cells);

        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_n_species(int n_species);

        ATLAS_HOST ATLAS_FORCE_INLINE int
        n_species() const;

        ATLAS_HOST ATLAS_FORCE_INLINE int
        n_pairs() const;

        ATLAS_HOST ATLAS_FORCE_INLINE int
        pair_index(int s, int r) const;

        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_sigma_t(int s, int r, T value);

        ATLAS_HOST ATLAS_FORCE_INLINE T
        sigma_t(int idx) const;

        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_number_weight(int s, T value);

        ATLAS_HOST ATLAS_FORCE_INLINE T
        number_weight(int s) const;

        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_cell_volume(T v);

        ATLAS_HOST ATLAS_FORCE_INLINE T
        cell_volume() const;

        ATLAS_HOST ATLAS_FORCE_INLINE T*
        sigma_t_pairs_ptr();

        ATLAS_HOST ATLAS_FORCE_INLINE T*
        number_weight_ptr();

    private:
        DeviceBuffer<T> d_g_ref_per_cell;
        DeviceBuffer<int> d_n_collisions;

        DeviceBuffer<int> d_n_particles_per_cell_species;

        DeviceBuffer<T> d_sigma_t_pairs;
        DeviceBuffer<T> d_number_weight_species;

        T _cell_volume = T(1);

        int _n_species = 1;
        int _n_pairs   = 1;
        int _n_cells   = 0;
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