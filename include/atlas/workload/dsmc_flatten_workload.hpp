#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>
#include <atlas/scan/exclusive_scan.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace atlas::workload {

template <typename T>
const atlas::DeviceBuffer<int>&
DsmcFlattenWorkload<T>::offsets() const noexcept {
    return collision_offsets;
}

template <typename T>
void
DsmcFlattenWorkload<T>::clear() {
    collision_offsets.resize(0);
    collision_cells.resize(0);
    flattened_collision_count = 0;
}

template <typename T>
bool
DsmcFlattenWorkload<T>::build(int* collision_count_ptr,
                               const int num_of_cells) {
    if (collision_offsets.size() != static_cast<std::size_t>(num_of_cells)) {
        collision_offsets.resize(static_cast<std::size_t>(num_of_cells));
    }

    atlas::exclusive_scan<atlas::ExecutionPolicy::device>(
        collision_count_ptr,
        collision_count_ptr + num_of_cells,
        collision_offsets.begin(),
        0);

    const auto last_cell  = static_cast<std::size_t>(num_of_cells - 1);
    const int last_offset = collision_offsets[last_cell];
    const int last_count  = collision_count_ptr[last_cell];

    if (last_offset > std::numeric_limits<int>::max() - last_count) {
        throw std::overflow_error("DsmcFlattenWorkload: flattened collision workload exceeds int range.");
    }

    const int total_collisions = last_offset + last_count;

    if (total_collisions <= 0) {
        collision_cells.resize(0);
        flattened_collision_count = 0;
        return false;
    }

    if (collision_cells.size() != static_cast<std::size_t>(total_collisions)) {
        collision_cells.resize(static_cast<std::size_t>(total_collisions));
    }

    flattened_collision_count = total_collisions;

    auto* collision_offsets_ptr = atlas::raw_pointer_cast(collision_offsets.data());
    auto* collision_cells_ptr   = atlas::raw_pointer_cast(collision_cells.data());

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int offset = collision_offsets_ptr[cell];
            const int count  = collision_count_ptr[cell];

            for (int local = 0; local < count; ++local) {
                collision_cells_ptr[offset + local] = cell;
            }
        });

    return true;
}

template <typename T>
void
DsmcFlattenWorkload<T>::schedule(const Probe& probe,
                                  const atlas::DeviceBuffer<int>*,
                                  const int) {
    if (!build(probe.collision_count_ptr, probe.num_of_cells)) {
        return;
    }

    const int* collision_offsets_ptr = atlas::raw_pointer_cast(collision_offsets.data());
    const int* collision_cells_ptr   = atlas::raw_pointer_cast(collision_cells.data());

    if (flattened_collision_count <= 0 || collision_offsets_ptr == nullptr || collision_cells_ptr == nullptr
        || probe.collision_count_ptr == nullptr || probe.properties_ptr == nullptr) {
        return;
    }

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        flattened_collision_count,
        [=] ATLAS_DEVICE(const int work_index) {
            const int cell            = collision_cells_ptr[work_index];
            const int local_collision = work_index - collision_offsets_ptr[cell];

            const int count     = static_cast<int>(probe.number_particle_ptr[cell]);
            const T max_sigma_g = probe.max_sigma_g_ptr[cell];

            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            DsmcFlattenWorkload<T>::execute_collision_trial(
                probe,
                cell,
                local_collision,
                begin,
                end,
                count,
                max_sigma_g);
        });
}

template <typename T>
bool
DsmcFlattenWorkload<T>::execute_collision_trial(const Probe& probe,
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
DsmcFlattenWorkload<T>::select_pair_offsets(const Probe& probe,
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
