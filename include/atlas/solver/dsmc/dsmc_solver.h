#pragma once

#include <atlas/core/macros.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/solver.h>

#include <cstdint>

namespace atlas::system {

enum class DsmcApplyMode : std::uint8_t {

    cell_sequential,

    flattened_atomic
};

enum class DsmcMajorantMode : std::uint8_t {

    exact_all_pairs,

    sampled_adaptive,

    sampled = sampled_adaptive
};

template <typename T>
struct DsmcSpeciesPairModel final {
    T coefficient {};
    T exponent {};
};

template <typename T>
class DsmcSolver : public Solver<T> {
public:
    class Builder;

public:
    struct DsmcSolverProbe {

        Vector3<T>* velocity_ptr {};

        const Vector3<T>* pre_collision_velocity_ptr {};

        const std::size_t* species_ptr {};

        const MaterialProperties<T>* properties_ptr {};

        const DsmcSpeciesPairModel<T>* species_pair_model_ptr {};

        T* number_particle_ptr {};

        T* max_relative_speed_ptr {};

        T* max_sigma_g_ptr {};

        int* collision_count_ptr {};

        const int* indices_ptr {};

        const int* cell_start_ptr {};

        const int* cell_end_ptr {};

        const int* allocated_solver_ptr {};

        int particle_count {};

        int num_of_cells {};

        int num_of_properties {};

        T cell_volume {};

        T statistical_weight {};

        DsmcKernelType kernel_type { DsmcKernelType::hard_sphere };

        DsmcKernel<T> kernel {};

        std::uint64_t collision_seed {};

        DsmcMajorantMode majorant_mode { DsmcMajorantMode::exact_all_pairs };

        int majorant_sample_count { 64 };

        T majorant_safety_factor { T(1) };

        T majorant_decay_factor { T(1) };

        const int* collision_offsets_ptr {};

        const int* flattened_collision_cells_ptr {};

        int flattened_collision_count {};

        int* particle_lock_ptr {};

        bool use_pre_collision_snapshot {};

        int* majorant_violation_count_ptr {};

        int* accepted_collision_count_ptr {};
    };

public:
    DsmcSolver() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    DsmcSolver(UniverseHostPtr<T> universe,
               FluidHostPtr<T> fluid,
               SpatialHashingSearcherHostPtr<T> searcher,
               DsmcKernelType kernel_type      = DsmcKernelType::hard_sphere,
               DsmcApplyMode apply_mode        = DsmcApplyMode::cell_sequential,
               DsmcMajorantMode majorant_mode  = DsmcMajorantMode::exact_all_pairs,
               int majorant_sample_count       = 64,
               T majorant_safety_factor        = T(1),
               T majorant_decay_factor         = T(1),
               bool use_pre_collision_snapshot = false,
               bool diagnostics_enabled        = false) noexcept;

    ~DsmcSolver() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(T dt) final;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(const DeviceBuffer<int>* allocated_solver, int index, T dt) final;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcKernelType
    kernel_type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcApplyMode
    apply_mode() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DsmcMajorantMode
    majorant_mode() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    majorant_sample_count() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    majorant_safety_factor() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    majorant_decay_factor() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    use_pre_collision_snapshot() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    diagnostics_enabled() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    majorant_violation_count() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    accepted_collision_count() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    invalidate_species_pair_models() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    collision_offsets() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<int>&
    flattened_collision_cells() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_universe_states();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_collision_data();

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build_collision_workload(const DsmcSolverProbe& probe, int index, T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    initialize_collision_context() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe(const DeviceBuffer<int>* allocated_solver, DsmcSolverProbe& probe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    measure_cell_collision_statistics(const DsmcSolverProbe& probe, int index, T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE bool
    build_flattened_collision_workload();

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    nth_valid_particle(int nth,
                       int begin,
                       int end,
                       int particle_count,
                       const int* indices_ptr) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    hashed_unit_interval(int index, std::uint64_t seed) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    hashed_index(int index, int upper_bound, std::uint64_t seed) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    sigma_g_from_pair_model(const DsmcSpeciesPairModel<T>& model, T relative_speed_squared) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    increment_diagnostic_counter(int* counter) noexcept;

    ATLAS_DEVICE ATLAS_FORCE_INLINE static void
    acquire_particle_pair_locks(int* locks, int particle_i, int particle_j) noexcept;

    ATLAS_DEVICE ATLAS_FORCE_INLINE static void
    release_particle_pair_locks(int* locks, int particle_i, int particle_j) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_cell_sequential_collisions(const DsmcSolverProbe& probe, int index, T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_collisions(const DsmcSolverProbe& probe, int index, T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_species_pair_models();

protected:
    DsmcKernel<T> _kernel {};

    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };

    DsmcApplyMode _apply_mode { DsmcApplyMode::cell_sequential };

    DsmcMajorantMode _majorant_mode { DsmcMajorantMode::exact_all_pairs };

    int _majorant_sample_count { 64 };

    T _majorant_safety_factor { T(1) };

    T _majorant_decay_factor { T(1) };

    bool _use_pre_collision_snapshot {};

    bool _diagnostics_enabled {};

    DeviceBuffer<int> _collision_offsets {};

    DeviceBuffer<int> _flattened_collision_cells {};

    DeviceBuffer<int> _particle_locks {};

    DeviceBuffer<Vector3<T>> _pre_collision_velocities {};

    DeviceBuffer<DsmcSpeciesPairModel<T>> _species_pair_models {};

    DeviceBuffer<int> _majorant_violation_count {};

    DeviceBuffer<int> _accepted_collision_count {};

    bool _species_pair_models_dirty { true };

    int _species_pair_model_property_count {};

    DsmcKernelType _species_pair_model_kernel_type { DsmcKernelType::hard_sphere };

    std::uint64_t _collision_seed = 0;
};

template <typename T>
class DsmcSolver<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_universe(UniverseHostPtr<T> universe) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_kernel_type(DsmcKernelType kernel_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_apply_mode(DsmcApplyMode apply_mode) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_majorant_mode(DsmcMajorantMode majorant_mode) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_majorant_sample_count(int majorant_sample_count) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_majorant_safety_factor(T majorant_safety_factor) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_majorant_decay_factor(T majorant_decay_factor) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_pre_collision_snapshot(bool use_pre_collision_snapshot) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_diagnostics_enabled(bool diagnostics_enabled) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DsmcSolver<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<DsmcSolver<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    UniverseHostPtr<T> _universe {};
    FluidHostPtr<T> _fluid {};
    SpatialHashingSearcherHostPtr<T> _searcher {};
    DsmcKernelType _kernel_type { DsmcKernelType::hard_sphere };
    DsmcApplyMode _apply_mode { DsmcApplyMode::cell_sequential };
    DsmcMajorantMode _majorant_mode { DsmcMajorantMode::exact_all_pairs };
    int _majorant_sample_count { 64 };
    T _majorant_safety_factor { T(1) };
    T _majorant_decay_factor { T(1) };
    bool _use_pre_collision_snapshot {};
    bool _diagnostics_enabled {};
};

}

namespace atlas {

using DsmcApplyMode = atlas::system::DsmcApplyMode;

using DsmcMajorantMode = atlas::system::DsmcMajorantMode;

template <typename T>
using DsmcSolver = atlas::system::DsmcSolver<T>;

template <typename T>
using DsmcSolverHostPtr = atlas::host_shared_ptr<atlas::system::DsmcSolver<T>>;

template <typename T>
using DsmcSolverDevicePtr = atlas::device_shared_ptr<atlas::system::DsmcSolver<T>>;

}

#include <atlas/solver/dsmc/dsmc_solver.hpp>