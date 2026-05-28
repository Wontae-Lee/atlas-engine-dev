#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>

#include <cstdint>

namespace atlas::workload {

template <typename T>
void
DsmcCellSequentialWorkload<T>::schedule(const Probe& probe,
                                        const atlas::DeviceBuffer<int>* allocated_solver,
                                        const int index) {
    if (probe.num_of_cells <= 0 || probe.collision_count_ptr == nullptr
        || probe.velocity_ptr == nullptr || probe.species_ptr == nullptr
        || probe.properties_ptr == nullptr
        || probe.indices_ptr == nullptr
        || probe.cell_start_ptr == nullptr || probe.cell_end_ptr == nullptr
        || probe.number_particle_ptr == nullptr || probe.max_sigma_g_ptr == nullptr) {
        return;
    }

    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int collisions = probe.collision_count_ptr[cell];
            const int count      = static_cast<int>(probe.number_particle_ptr[cell]);
            const T max_sigma_g  = probe.max_sigma_g_ptr[cell];

            if (collisions <= 0 || count < 2 || !(max_sigma_g > T(0))) {
                return;
            }

            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (begin < 0 || end <= begin) {
                return;
            }

            for (int local_collision = 0; local_collision < collisions; ++local_collision) {
                DsmcCellSequentialWorkload<T>::execute_collision_trial(
                    probe,
                    cell,
                    local_collision,
                    begin,
                    end,
                    count,
                    max_sigma_g);
            }
        });
}

template <typename T>
bool
DsmcCellSequentialWorkload<T>::execute_collision_trial(const Probe& probe,
                                                       const int cell,
                                                       const int local_collision,
                                                       const int begin,
                                                       const int end,
                                                       const int count,
                                                       const T max_sigma_g) noexcept {
    if (count < 2 || local_collision < 0 || begin < 0 || end <= begin || !(max_sigma_g > T(0))) {
        return false;
    }

    int lhs_local = -1;
    int rhs_local = -1;

    if (!select_pair_offsets(probe, lhs_local, rhs_local, cell, local_collision, count)) {
        return false;
    }

    return DsmcCollisionWorkload<T>::execute_collision_pair(
        probe,
        cell,
        local_collision,
        begin,
        end,
        lhs_local,
        rhs_local,
        max_sigma_g);
}

template <typename T>
bool
DsmcCellSequentialWorkload<T>::select_pair_offsets(const Probe& probe,
                                                   int& lhs_local,
                                                   int& rhs_local,
                                                   const int cell,
                                                   const int local_collision,
                                                   const int count) noexcept {
    const auto stream = static_cast<std::uint64_t>(cell) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
        + static_cast<std::uint64_t>(local_collision);

    lhs_local = atlas::sampling::sample_hashed_index(
        cell,
        count,
        probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_LHS_SALT);

    rhs_local = atlas::sampling::sample_hashed_index(
        cell,
        count - 1,
        probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_RHS_SALT);

    if (rhs_local >= lhs_local) {
        ++rhs_local;
    }

    return true;
}

}
