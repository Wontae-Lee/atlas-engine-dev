#pragma once

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/solver.h>

#include <cstdint>

namespace atlas::system {

template <typename T>
class DsmcSolver : public Solver<T> {
public:
    struct DsmcSolverProbe {
        Vector3<T>* velocity_ptr {};
        const std::size_t* species_ptr {};
        const MaterialProperties<T>* properties_ptr {};

        T* number_particle_ptr {};
        T* max_relative_speed_ptr {};
        T* max_sigma_g_ptr {};
        int* collision_count_ptr {};

        const int* indices_ptr {};
        const int* cell_start_ptr {};
        const int* cell_end_ptr {};

        int particle_count {};
        int num_of_cells {};

        T cell_volume {};
        T statistical_weight {};

        DsmcKernel<T> kernel {};
        std::uint64_t collision_seed {};
    };

public:
    DsmcSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcSolver(UniverseHostPtr<T> universe,
               FluidHostPtr<T> fluid,
               SpatialHashingSearcherHostPtr<T> searcher,
               DsmcKernelType kernel_type = DsmcKernelType::hard_sphere) noexcept;

    ~DsmcSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) override;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) override;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcKernelType
    kernel_type() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_states();

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    reset_states();

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    measure_cell_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, T dt);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    nth_valid_particle(int nth,
                       int begin,
                       int end,
                       int particle_count,
                       const int* indices_ptr) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    sigma_g(const DsmcKernel<T>& kernel,
            const MaterialProperties<T>* properties_ptr,
            std::size_t species_i,
            std::size_t species_j,
            T relative_speed_squared) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, T dt) = 0;

protected:
    UniverseHostPtr<T> _universe {};
    FluidHostPtr<T> _fluid {};
    SpatialHashingSearcherHostPtr<T> _searcher {};

    DsmcSolverProbe _probe {};
    DsmcKernel<T> _kernel {};
    std::uint64_t _collision_seed = 0;
};

}

namespace atlas {

template <typename T>
using DsmcSolver = atlas::system::DsmcSolver<T>;

template <typename T>
using DsmcSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcSolver<T>>;

template <typename T>
using DsmcSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcSolver<T>>;

}

#include <atlas/solver/dsmc/dsmc_solver.hpp>
