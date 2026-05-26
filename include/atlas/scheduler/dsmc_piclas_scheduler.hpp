#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>

namespace atlas::scheduler {

template <typename T>
void
DsmcPiclasScheduler<T>::schedule(const Probe& probe,
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
                DsmcPiclasScheduler<T>::execute_collision_trial(
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
DsmcPiclasScheduler<T>::limits_collision_count() const noexcept {
    return true;
}

template <typename T>
void
DsmcPiclasScheduler<T>::select_pair_offsets(int& lhs_local,
                                            int& rhs_local,
                                            const int local_collision,
                                            const int count,
                                            const int selector,
                                            const std::uint64_t seed) noexcept {
    lhs_local = -1;
    rhs_local = -1;

    if (count < 2 || local_collision < 0 || local_collision >= count / 2) {
        return;
    }

    const int offset = atlas::sampling::sample_hashed_index(
        selector,
        count,
        seed + atlas::seed::DSMC_COLLISION_LHS_SALT);

    int stride = count == 2
        ? 1
        : 1 + atlas::sampling::sample_hashed_index(
              selector,
              count - 1,
              seed + atlas::seed::DSMC_COLLISION_RHS_SALT);

    for (;;) {
        int a = stride;
        int b = count;
        while (b != 0) {
            const int next = a % b;
            a = b;
            b = next;
        }

        if (a == 1) {
            break;
        }

        ++stride;
        if (stride >= count) {
            stride = 1;
        }
    }

    const auto lhs_offset = static_cast<std::int64_t>(2 * local_collision) * static_cast<std::int64_t>(stride);
    const auto rhs_offset = static_cast<std::int64_t>(2 * local_collision + 1) * static_cast<std::int64_t>(stride);
    lhs_local = static_cast<int>((static_cast<std::int64_t>(offset) + lhs_offset) % count);
    rhs_local = static_cast<int>((static_cast<std::int64_t>(offset) + rhs_offset) % count);
}

template <typename T>
bool
DsmcPiclasScheduler<T>::execute_collision_trial(const Probe& probe,
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
    select_pair_offsets(
        lhs_local,
        rhs_local,
        local_collision,
        count,
        cell,
        probe.collision_seed + static_cast<std::uint64_t>(cell));

    if (lhs_local < 0 || rhs_local < 0) {
        return false;
    }

    return DsmcCollisionScheduler<T>::execute_collision_pair(
        probe,
        cell,
        local_collision,
        begin,
        end,
        lhs_local,
        rhs_local,
        max_sigma_g);
}

}
