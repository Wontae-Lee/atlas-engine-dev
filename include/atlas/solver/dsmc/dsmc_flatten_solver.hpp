#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>
#include <atlas/scan/exclusive_scan.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
typename DsmcFlattenSolver<T>::Builder
DsmcFlattenSolver<T>::builder() noexcept {

    // Return a fresh builder so callers can configure dependencies fluently.
    return Builder {};
}

template <typename T>
const atlas::DeviceBuffer<int>&
DsmcFlattenSolver<T>::offsets() const noexcept {

    // Expose the exclusive prefix offsets used by the flattened collision queue.
    return collision_offsets;
}

template <typename T>
void
DsmcFlattenSolver<T>::clear() {

    // Release the current flattened collision schedule.
    collision_offsets.resize(0);
    collision_cells.resize(0);
    filtered_collision_counts.resize(0);

    // Reset the total number of scheduled collision work items.
    flattened_collision_count = 0;
}

template <typename T>
bool
DsmcFlattenSolver<T>::build(int* collision_count_ptr,
                            const int num_of_cells,
                            const int* allocated_solver_ptr,
                            const int index) {
    // If the input parameters are invalid, clear the solver and return failure.
    if (num_of_cells <= 0 || collision_count_ptr == nullptr) {
        clear();
        return false;
    }

    // One exclusive offset is required for each simulation cell.
    if (collision_offsets.size() != static_cast<std::size_t>(num_of_cells)) {
        collision_offsets.resize(static_cast<std::size_t>(num_of_cells));
    }

    int* filtered_collision_count_ptr = nullptr;

    if (allocated_solver_ptr != nullptr) {
        // Allocate a filtered count buffer when only a subset of cells belongs to this solver.
        if (filtered_collision_counts.size() != static_cast<std::size_t>(num_of_cells)) {
            filtered_collision_counts.resize(static_cast<std::size_t>(num_of_cells));
        }

        filtered_collision_count_ptr = atlas::raw_pointer_cast(filtered_collision_counts.data());

        // Copy collision counts only for cells assigned to the requested solver index.
        atlas::parallel_for<atlas::ExecutionPolicy::device>(
            0,
            num_of_cells,
            [=] ATLAS_DEVICE(const int cell) {
                filtered_collision_count_ptr[cell] = allocated_solver_ptr[cell] == index
                    ? collision_count_ptr[cell]
                    : 0;
            });
    }

    // Use filtered counts when solver ownership is provided; otherwise use raw counts.
    int* scheduled_collision_count_ptr = filtered_collision_count_ptr != nullptr
        ? filtered_collision_count_ptr
        : collision_count_ptr;

    // Convert per-cell collision counts into exclusive offsets.
    atlas::exclusive_scan<atlas::ExecutionPolicy::device>(
        scheduled_collision_count_ptr,
        scheduled_collision_count_ptr + num_of_cells,
        collision_offsets.begin(),
        0);

    // The total count is the last exclusive offset plus the last cell count.
    const auto last_cell  = static_cast<std::size_t>(num_of_cells - 1);
    const int last_offset = collision_offsets[last_cell];
    const int last_count  = scheduled_collision_count_ptr[last_cell];

    // Guard against overflowing the integer work-index range.
    if (last_offset > std::numeric_limits<int>::max() - last_count) {
        throw std::overflow_error("DsmcFlattenSolver: flattened collision solver exceeds int range.");
    }

    const int total_collisions = last_offset + last_count;

    // No collision work is scheduled when all cell counts are zero.
    if (total_collisions <= 0) {
        collision_cells.resize(0);
        flattened_collision_count = 0;
        return false;
    }

    // Allocate one cell-index entry for every flattened collision work item.
    if (collision_cells.size() != static_cast<std::size_t>(total_collisions)) {
        collision_cells.resize(static_cast<std::size_t>(total_collisions));
    }

    flattened_collision_count = total_collisions;

    auto* collision_offsets_ptr = atlas::raw_pointer_cast(collision_offsets.data());
    auto* collision_cells_ptr   = atlas::raw_pointer_cast(collision_cells.data());

    // Fill the flattened work-to-cell mapping.
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int offset = collision_offsets_ptr[cell];
            const int count  = scheduled_collision_count_ptr[cell];

            // Each scheduled collision slot for this cell stores the owning cell index.
            for (int local = 0; local < count; ++local) {
                collision_cells_ptr[offset + local] = cell;
            }
        });

    return true;
}

template <typename T>
void
DsmcFlattenSolver<T>::apply_collision(const atlas::DeviceBuffer<int>* allocated_solver,
                                      const int index,
                                      const T) {

    // Copy the current DSMC probe so the device lambda captures stable raw pointers.
    const auto probe = this->_probe;

    // Optional per-cell solver ownership map.
    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    // Build the flattened work queue from the measured per-cell collision counts.
    if (!build(probe.collision_count_ptr, probe.num_of_cells, allocated_solver_ptr, index)) {
        return;
    }

    const int* collision_offsets_ptr = atlas::raw_pointer_cast(collision_offsets.data());
    const int* collision_cells_ptr   = atlas::raw_pointer_cast(collision_cells.data());

    // Defensive guard against invalid or empty collision schedules.
    if (flattened_collision_count <= 0 || collision_offsets_ptr == nullptr || collision_cells_ptr == nullptr
        || probe.collision_count_ptr == nullptr || probe.properties_ptr == nullptr) {
        return;
    }

    // Launch one device work item per scheduled collision attempt.
    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        flattened_collision_count,
        [=] ATLAS_DEVICE(const int work_index) {
            // Map the flattened work index back to its owning cell.
            const int cell = collision_cells_ptr[work_index];

            // Recover the local collision index inside the owning cell.
            const int local_collision = work_index - collision_offsets_ptr[cell];

            const int count     = static_cast<int>(probe.number_particle_ptr[cell]);
            const T max_sigma_g = probe.max_sigma_g_ptr[cell];

            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            // Build a deterministic random stream for this cell-local collision attempt.
            const auto stream = static_cast<std::uint64_t>(cell) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
                + static_cast<std::uint64_t>(local_collision);

            // Sample the first local particle index uniformly from the cell.
            const int lhs_local = atlas::sampling::sample_hashed_index(
                cell,
                count,
                probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_LHS_SALT);

            // Sample the second local particle index from one fewer slot.
            int rhs_local = atlas::sampling::sample_hashed_index(
                cell,
                count - 1,
                probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_RHS_SALT);

            // Shift the sampled second index to guarantee lhs_local != rhs_local.
            if (rhs_local >= lhs_local) {
                ++rhs_local;
            }

            // Perform the DSMC acceptance test and velocity update for this pair.
            DsmcFlattenSolver<T>::collide_pair(
                probe,
                cell,
                local_collision,
                begin,
                end,
                lhs_local,
                rhs_local,
                max_sigma_g);
        });
}

template <typename T>
typename DsmcFlattenSolver<T>::Builder&
DsmcFlattenSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {

    // Store the universe dependency for later solver construction.
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename DsmcFlattenSolver<T>::Builder&
DsmcFlattenSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {

    // Store the fluid dependency for later solver construction.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcFlattenSolver<T>::Builder&
DsmcFlattenSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {

    // Store the spatial searcher dependency for later solver construction.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename DsmcFlattenSolver<T>::Builder&
DsmcFlattenSolver<T>::Builder::with_kernel_type(const atlas::system::DsmcKernelType kernel_type) noexcept {

    // Select the DSMC collision model used by the constructed solver.
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
void
DsmcFlattenSolver<T>::Builder::validate() const {

    // A solver cannot be constructed without a universe.
    if (!_universe) {
        throw std::runtime_error("DsmcFlattenSolver::Builder: universe must not be null.");
    }

    // A solver cannot be constructed without fluid particle storage.
    if (!_fluid) {
        throw std::runtime_error("DsmcFlattenSolver::Builder: fluid must not be null.");
    }

    // A solver cannot be constructed without a spatial hashing searcher.
    if (!_searcher) {
        throw std::runtime_error("DsmcFlattenSolver::Builder: searcher must not be null.");
    }
}

template <typename T>
DsmcFlattenSolver<T>
DsmcFlattenSolver<T>::Builder::build() const {

    // Validate required dependencies before constructing the solver value.
    validate();

    return DsmcFlattenSolver<T>(_universe, _fluid, _searcher, _kernel_type);
}

template <typename T>
atlas::host_shared_ptr<DsmcFlattenSolver<T>>
DsmcFlattenSolver<T>::Builder::make_host_shared() const {

    // Validate required dependencies before constructing the shared solver object.
    validate();

    return atlas::make_host_shared<DsmcFlattenSolver<T>>(_universe, _fluid, _searcher, _kernel_type);
}

}