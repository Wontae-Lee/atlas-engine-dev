#pragma once

#include <atlas/core/macros.h>
#include <atlas/random/seed.h>
#include <atlas/sampling/sampling.h>
#include <atlas/solver/dsmc/dsmc_flatten_workload.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/dsmc/dsmc_probe.h>
#include <atlas/solver/dsmc/statistics/dsmc_statistics.h>
#include <atlas/solver/solver.h>

#include <cstdint>

namespace atlas {

enum struct DsmcCollisionWorkloadType : int {

    cell,

    flatten
};

class DsmcSolver : public Solver {
public:
    using Probe = atlas::DsmcProbe;

    DsmcSolver() = default;

    ATLAS_HOST
    DsmcSolver(UniverseHostPtr universe,
               FluidHostPtr fluid,
               SearcherHostPtr searcher,
               DsmcKernelType kernel_type              = DsmcKernelType::hard_sphere,
               DsmcCollisionWorkloadType workload_type = DsmcCollisionWorkloadType::cell) noexcept;

    ~DsmcSolver() override = default;

    ATLAS_HOST void
    solve(float dt) override;

    ATLAS_HOST void
    solve(const DeviceBuffer<int>* allocated_solver, int index, float dt) override;

    ATLAS_NODISCARD ATLAS_HOST DsmcKernelType
    kernel_type() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST DsmcCollisionWorkloadType
    workload_type() const noexcept;

    ATLAS_HOST void
    set_workload_type(DsmcCollisionWorkloadType workload_type) noexcept;

    ATLAS_HOST void
    ensure_states();

    ATLAS_HOST virtual void
    reset_states();

    ATLAS_HOST void
    make_probe() noexcept;

    ATLAS_HOST virtual bool
    measure_collision_statistics(const DeviceBuffer<int>* allocated_solver, int index, float dt);

    ATLAS_HOST virtual void
    apply_collision(const DeviceBuffer<int>* allocated_solver, int index, float dt);

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_pair(const Probe& probe,
                 int cell,
                 int local_collision,
                 int begin,
                 int end,
                 int lhs_local,
                 int rhs_local,
                 float max_sigma_g) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static bool
    collide_indexed_pair(const Probe& probe,
                         int cell,
                         std::uint64_t stream,
                         int particle_i,
                         int particle_j,
                         float max_sigma_g) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    sample_distinct_pair(int& lhs_local,
                         int& rhs_local,
                         int cell,
                         int count,
                         std::uint64_t seed,
                         std::uint64_t stream) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static int
    particle_at(int nth,
                int begin,
                int end,
                int particle_count,
                const int* indices_ptr) noexcept;

protected:
    ATLAS_HOST virtual void
    apply_flattened_collision(const DeviceBuffer<int>* allocated_solver, int index);

    DsmcProbe _probe {};

    DsmcKernel _kernel {};

    DsmcCollisionWorkloadType _workload_type { DsmcCollisionWorkloadType::cell };

    DsmcFlattenWorkload _flatten_workload {};

    DsmcStatistics _statistics {};

    std::uint64_t _collision_seed = 0;

public:
    ATLAS_HOST void
    apply_cell_collisions(const int* allocated_solver_ptr, int index);

    ATLAS_HOST void
    launch_flattened_collisions();
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
DsmcSolver::collide_pair(const Probe& probe,
                         const int cell,
                         const int local_collision,
                         const int begin,
                         const int end,
                         const int lhs_local,
                         const int rhs_local,
                         const float max_sigma_g) noexcept {
    const int particle_i = particle_at(
        lhs_local,
        begin,
        end,
        probe.particle_count,
        probe.indices_ptr);

    const int particle_j = particle_at(
        rhs_local,
        begin,
        end,
        probe.particle_count,
        probe.indices_ptr);

    const auto stream = static_cast<std::uint64_t>(cell) * atlas::DSMC_CELL_STREAM_MULTIPLIER
        + static_cast<std::uint64_t>(local_collision);
    return collide_indexed_pair(
        probe,
        cell,
        stream,
        particle_i,
        particle_j,
        max_sigma_g);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
DsmcSolver::collide_indexed_pair(const Probe& probe,
                                 const int cell,
                                 const std::uint64_t stream,
                                 const int particle_i,
                                 const int particle_j,
                                 const float max_sigma_g) noexcept {

    if (particle_i < 0 || particle_j < 0 || particle_i == particle_j) {
        return false;
    }

    const std::size_t species_i = probe.species_ptr[particle_i];
    const std::size_t species_j = probe.species_ptr[particle_j];
    if (species_i >= static_cast<std::size_t>(probe.species_count)
        || species_j >= static_cast<std::size_t>(probe.species_count)) {
        return false;
    }

    Float3 lhs_velocity = probe.velocity_ptr[particle_i];
    Float3 rhs_velocity = probe.velocity_ptr[particle_j];

    const float relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
    const float sigma_g                = probe.kernel.sigma_g(
        probe.properties_ptr,
        species_i,
        species_j,
        relative_speed_squared);

    if (!(sigma_g > 0.0f)) {
        return false;
    }

    float local_max_sigma_g = max_sigma_g;
    if (sigma_g > local_max_sigma_g) {
        local_max_sigma_g           = sigma_g;
        probe.max_sigma_g_ptr[cell] = sigma_g;
    }

    float accept_probability = sigma_g / local_max_sigma_g;
    if (accept_probability > 1.0f) {
        accept_probability = 1.0f;
    }

    const float accept_sample = atlas::sample_hashed_unit_interval(
        cell,
        probe.collision_seed + stream + atlas::DSMC_COLLISION_ACCEPT_SALT);

    if (accept_sample >= accept_probability) {
        return false;
    }

    probe.kernel(
        lhs_velocity,
        rhs_velocity,
        probe.properties_ptr[species_i],
        probe.properties_ptr[species_j]);

    probe.velocity_ptr[particle_i] = lhs_velocity;
    probe.velocity_ptr[particle_j] = rhs_velocity;

    return true;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
DsmcSolver::sample_distinct_pair(int& lhs_local,
                                 int& rhs_local,
                                 const int cell,
                                 const int count,
                                 const std::uint64_t seed,
                                 const std::uint64_t stream) noexcept {
    lhs_local = atlas::sample_hashed_index(
        cell,
        count,
        seed + stream + atlas::DSMC_COLLISION_LHS_SALT);

    rhs_local = atlas::sample_hashed_index(
        cell,
        count - 1,
        seed + stream + atlas::DSMC_COLLISION_RHS_SALT);

    if (rhs_local >= lhs_local) {
        ++rhs_local;
    }
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE int
DsmcSolver::particle_at(const int nth,
                        const int begin,
                        const int end,
                        const int particle_count,
                        const int* indices_ptr) noexcept {

    const int sorted_index = begin + nth;

    if (nth < 0 || sorted_index < begin || sorted_index >= end) {
        return -1;
    }

    const int particle_index = indices_ptr[sorted_index];

    return (particle_index >= 0 && particle_index < particle_count) ? particle_index : -1;
}

using DsmcSolverHostPtr = atlas::host_shared_ptr<atlas::DsmcSolver>;

using DsmcSolverDevicePtr = atlas::device_shared_ptr<atlas::DsmcSolver>;

}
