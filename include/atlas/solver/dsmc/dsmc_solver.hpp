#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>

#include <cmath>
#include <stdexcept>

namespace atlas::system {

template <typename T>
DsmcSolver<T>::DsmcSolver(UniverseHostPtr<T> universe,
                          FluidHostPtr<T> fluid,
                          SpatialHashingSearcherHostPtr<T> searcher,
                          const DsmcKernelType kernel_type) noexcept
    // Forward the shared simulation dependencies to the base Solver class.
    //
    // The base class owns the common solver-level references to:
    //   - the universe, which provides cell topology and universe states,
    //   - the fluid, which provides particle states and material properties,
    //   - the spatial searcher, which organizes particles by cell.
    //
    // This DSMC solver extends that shared infrastructure with DSMC-specific
    // collision-kernel state and per-cell collision workload buffers.
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher))

    // Construct the tagged collision kernel object from the selected kernel type.
    //
    // The DsmcKernel wrapper stores the active collision model implementation
    // and provides a uniform dispatch interface at runtime.
    , _kernel(DsmcKernel<T>(kernel_type))

    // Store the selected kernel type explicitly so later code can reuse the
    // enum value directly, for example when evaluating cross sections.
    , _kernel_type(kernel_type) {
    // Ensure that all universe-side state buffers required by the DSMC solver
    // exist immediately after construction.
    //
    // This makes the solver robust against partially prepared universes by
    // lazily allocating missing states on first construction.
    ensure_universe_states();
}

template <typename T>
void
DsmcSolver<T>::solve(const T dt) {
    // Dispatch to the partition-aware overload without solver partitioning.
    //
    // Passing:
    //   - nullptr as the allocation map means all cells are eligible,
    //   - 0 as the solver index is a dummy value in this mode.
    solve(nullptr, 0, dt);
}

template <typename T>
void
DsmcSolver<T>::solve(const DeviceBuffer<int>* allocated_solver, const int index, const T dt) {
    // Build all per-cell collision statistics required for the upcoming
    // collision application phase.
    //
    // This stage validates the solver context, rebuilds the spatial search
    // structure, measures per-cell particle and relative-speed statistics,
    // and computes how many collision attempts should be processed per cell.
    if (!build_collision_workload(allocated_solver, index, dt)) {
        // Exit early when collision workload construction fails or when
        // collision processing is not meaningful under the current conditions.
        return;
    }

    // Apply the actual particle collisions using the workload produced above.
    //
    // At this point, per-cell collision counts have already been computed and
    // the derived solver implementation can execute its collision strategy.
    apply_collisions(allocated_solver, index, dt);
}

template <typename T>
void
DsmcSolver<T>::ensure_universe_states() {
    // Nothing can be initialized when the universe dependency is absent.
    //
    // The solver is designed to fail gracefully in that situation and defer
    // meaningful work until a valid universe is provided.
    if (!this->_universe) {
        return;
    }

    // Cache the total number of cells and convert it to a size-compatible type
    // for device-buffer allocation and resizing operations.
    const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());

    // Ensure the universe owns a per-cell particle-count state.
    //
    // This state records how many particles belong to each cell during the
    // current collision preparation stage.
    if (!this->_universe->template has_state<atlas::universe::UniverseNumberParticleState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseNumberParticleState<T>>(number_of_cells);
    }

    // Ensure the universe owns a per-cell maximum-relative-speed state.
    //
    // This state stores the largest pairwise relative speed observed within
    // each cell during collision statistics measurement.
    if (!this->_universe->template has_state<atlas::universe::UniverseMaxRelativeSpeedState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseMaxRelativeSpeedState<T>>(number_of_cells);
    }

    // Ensure the universe owns a per-cell collision-count state.
    //
    // This state stores the number of collision attempts that will be executed
    // for each cell during the current solve step.
    if (!this->_universe->template has_state<atlas::universe::UniverseCollisionCountState<int>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseCollisionCountState<int>>(number_of_cells);
    }

    // Keep the internal offset buffer aligned with the current cell count.
    //
    // This buffer is later used to build a flattened representation of the
    // collision workload, where each collision attempt maps to a source cell.
    _collision_offsets.resize(number_of_cells);
}

template <typename T>
void
DsmcSolver<T>::reset_collision_data() {
    // When no universe exists, all collision-side buffers must be reset to an
    // empty state because no valid per-cell information can be maintained.
    if (!this->_universe) {
        _collision_offsets.resize(0);
        _flattened_collision_cells.resize(0);
        return;
    }

    // Read the current number of cells from the universe.
    const auto num_of_cells = this->_universe->number_of_cells();

    // If the universe defines no cells, clear all solver-owned collision data.
    //
    // In this case there is no valid domain over which cell-based collision
    // statistics could be computed.
    if (num_of_cells <= 0) {
        _collision_offsets.resize(0);
        _flattened_collision_cells.resize(0);
        return;
    }

    // Ensure all required universe states exist before attempting to clear them.
    ensure_universe_states();

    // Retrieve the per-cell particle-count state.
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();

    // Retrieve the per-cell maximum-relative-speed state.
    auto* max_relative_speed_state
        = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();

    // Retrieve the per-cell collision-count state.
    auto* collision_count_state
        = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    // Reset the particle-count state to zero for every cell.
    //
    // This clears stale values from previous solve steps or partially completed
    // initialization attempts.
    if (number_particle_state != nullptr) {
        auto& buffer = number_particle_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), T(0));
    }

    // Reset the maximum-relative-speed state to zero for every cell.
    if (max_relative_speed_state != nullptr) {
        auto& buffer = max_relative_speed_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), T(0));
    }

    // Reset the collision-count state to zero for every cell.
    if (collision_count_state != nullptr) {
        auto& buffer = collision_count_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), 0);
    }

    // Ensure the internal collision-offset buffer matches the current cell count.
    if (_collision_offsets.size() != static_cast<std::size_t>(num_of_cells)) {
        _collision_offsets.resize(static_cast<std::size_t>(num_of_cells));
    }

    // Reset all offsets to zero because no flattened workload currently exists.
    atlas::parallel_fill<ExecutionPolicy::device>(_collision_offsets.begin(), _collision_offsets.end(), 0);

    // Drop the flattened collision-to-cell mapping completely.
    _flattened_collision_cells.resize(0);
}

template <typename T>
bool
DsmcSolver<T>::build_collision_workload(const DeviceBuffer<int>* allocated_solver,
                                        const int index,
                                        const T dt) {
    // Initialize and validate the collision-processing context.
    //
    // This step ensures that:
    //   - all solver dependencies are available,
    //   - universe-side states exist,
    //   - the spatial search structure is built,
    //   - required particle and universe states are present.
    if (!initialize_collision_context()) {
        return false;
    }

    // The time step must be strictly positive for a meaningful DSMC collision
    // estimate because the expected number of collisions scales with dt.
    if (!(dt > T(0))) {
        throw std::invalid_argument("DsmcSolver: dt must be positive.");
    }

    // Measure per-cell collision statistics and populate:
    //   - number of particles per cell,
    //   - maximum relative speed per cell,
    //   - number of collision attempts per cell.
    if (!measure_cell_collision_statistics(allocated_solver, index, dt)) {
        return false;
    }

    // At this stage the collision workload stored in universe states is valid.
    return true;
}

template <typename T>
bool
DsmcSolver<T>::initialize_collision_context() noexcept {
    // All three shared dependencies are required for collision preparation.
    //
    // Without:
    //   - a universe, there is no cell domain,
    //   - a fluid, there are no particles or material properties,
    //   - a searcher, particles cannot be grouped by cell.
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        reset_collision_data();
        return false;
    }

    // Ensure that all required universe states exist before proceeding.
    ensure_universe_states();

    // Rebuild the spatial search structure so the cell-to-particle mapping
    // reflects the latest particle positions before collision statistics are
    // measured.
    this->_searcher->build();

    // Retrieve the velocity state required for relative-speed computation.
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();

    // Retrieve the species state required for material-property lookup.
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    // Retrieve the per-cell particle-count state.
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();

    // Retrieve the per-cell maximum-relative-speed state.
    auto* max_relative_speed_state
        = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();

    // Retrieve the per-cell collision-count state.
    auto* collision_count_state
        = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    // All of these states are mandatory for subsequent device-side statistics
    // evaluation. Reset internal data and fail safely if any are missing.
    if (velocity_state == nullptr || species_state == nullptr || number_particle_state == nullptr
        || max_relative_speed_state == nullptr || collision_count_state == nullptr) {
        reset_collision_data();
        return false;
    }

    return true;
}

template <typename T>
bool
DsmcSolver<T>::measure_cell_collision_statistics(const DeviceBuffer<int>* allocated_solver,
                                                 const int index,
                                                 const T dt) {
    // Reacquire the particle and universe states required to compute collision
    // statistics for the current step.
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state
        = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();
    auto* collision_count_state
        = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    // Cache references to the underlying state buffers for compact and readable
    // access in the rest of the function.
    auto& velocities          = velocity_state->data();
    auto& particle_species    = species_state->data();
    auto& number_particle     = number_particle_state->data();
    auto& max_relative_speed  = max_relative_speed_state->data();
    auto& collision_count     = collision_count_state->data();

    // Retrieve the species-specific material-property table used by the
    // collision kernel cross-section evaluation.
    auto& particle_properties = this->_fluid->particle_properties();

    // Convert buffer-backed containers to raw pointers suitable for device
    // lambda capture and device-side indexing.
    const auto* velocity_ptr     = atlas::raw_pointer_cast(velocities.data());
    const auto* species_ptr      = atlas::raw_pointer_cast(particle_species.data());
    auto* number_particle_ptr    = atlas::raw_pointer_cast(number_particle.data());
    auto* max_relative_speed_ptr = atlas::raw_pointer_cast(max_relative_speed.data());
    auto* collision_count_ptr    = atlas::raw_pointer_cast(collision_count.data());
    const auto* properties_ptr   = atlas::raw_pointer_cast(particle_properties.data());

    // Retrieve the spatial-searcher arrays that define the cell-local particle
    // layout used during per-cell pair enumeration.
    const auto* indices_ptr      = this->_searcher->indices();
    const auto* cell_start_ptr   = this->_searcher->cell_start();
    const auto* cell_end_ptr     = this->_searcher->cell_end();

    // Cache scalar quantities used repeatedly inside the device kernel.
    const int particle_count     = static_cast<int>(this->_fluid->particle_count());
    const int num_of_cells       = this->_universe->number_of_cells();
    const int num_of_properties  = static_cast<int>(particle_properties.size());
    const T cell_volume          = this->_universe->cell_volume();
    const T statistical_weight   = this->_fluid->statistical_weight();

    // Cache the selected kernel type because the static cross-section dispatch
    // below depends on the collision model in use.
    const DsmcKernelType kernel_type = _kernel_type;

    // Abort when the global simulation context makes collision measurement
    // impossible or meaningless.
    //
    // Cases handled here:
    //   - fewer than two particles exist globally,
    //   - no cells exist,
    //   - no material properties are available,
    //   - the cell volume is non-positive.
    if (particle_count < 2 || num_of_cells <= 0 || num_of_properties <= 0 || !(cell_volume > T(0))) {
        reset_collision_data();
        return false;
    }

    // Launch one device work item per cell to measure local collision statistics.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=, allocated_solver_ptr = allocated_solver != nullptr
                ? atlas::raw_pointer_cast(allocated_solver->data())
                : nullptr] ATLAS_DEVICE(const int cell) {
            // Respect optional external cell partitioning.
            //
            // When an allocation map is provided, only the cells assigned to the
            // current solver index participate in this solve call. Cells assigned
            // elsewhere are reset to zero so their local statistics are known to
            // be inactive for this solver instance.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                number_particle_ptr[cell]    = T(0);
                max_relative_speed_ptr[cell] = T(0);
                collision_count_ptr[cell]    = 0;
                return;
            }

            // Read the contiguous searcher range for the current cell.
            const int begin = cell_start_ptr[cell];
            const int end   = cell_end_ptr[cell];

            // Handle empty or invalid cell ranges defensively.
            //
            // Such cells contain no valid particle span for pair enumeration.
            if (begin < 0 || end <= begin) {
                number_particle_ptr[cell]    = T(0);
                max_relative_speed_ptr[cell] = T(0);
                collision_count_ptr[cell]    = 0;
                return;
            }

            // Infer the number of particles associated with the cell from the
            // searcher range length.
            const int count = end - begin;

            // Track the largest observed relative speed in the cell.
            T max_relative = T(0);

            // Track the maximum value of sigma * g over all candidate particle
            // pairs in the cell, where:
            //   - sigma is the collision cross section,
            //   - g is the relative speed.
            //
            // This quantity is used by the NTC estimate of how many collision
            // attempts should be scheduled in the cell.
            T max_sigma_g  = T(0);

            // Enumerate all unordered particle pairs in the cell.
            //
            // This is an O(n^2) local scan over the particles in the current
            // cell, which is acceptable because the scan is confined to each
            // cell independently.
            for (int a = begin; a < end; ++a) {
                const int particle_i = indices_ptr[a];

                for (int b = a + 1; b < end; ++b) {
                    const int particle_j = indices_ptr[b];

                    // Read the species identifiers of the two particles.
                    const std::size_t species_i = species_ptr[particle_i];
                    const std::size_t species_j = species_ptr[particle_j];

                    // Skip the pair if either species index is outside the valid
                    // material-property table range.
                    if (species_i >= static_cast<std::size_t>(num_of_properties)
                        || species_j >= static_cast<std::size_t>(num_of_properties)) {
                        continue;
                    }

                    // Compute the relative velocity vector of the pair.
                    const Vector3<T> relative_velocity = velocity_ptr[particle_i] - velocity_ptr[particle_j];

                    // Compute the relative speed magnitude.
                    const T relative_speed             = relative_velocity.length();

                    // Update the per-cell maximum relative speed if this pair
                    // exceeds the previously observed maximum.
                    if (relative_speed > max_relative) {
                        max_relative = relative_speed;
                    }

                    // Evaluate the collision cross section using the selected
                    // kernel model and the material properties of both species.
                    const T sigma = DsmcKernel<T>::cross_section(
                        kernel_type,
                        properties_ptr[species_i],
                        properties_ptr[species_j],
                        relative_speed);

                    // Compute sigma * g for this pair.
                    const T sigma_g = sigma * relative_speed;

                    // Track the maximum sigma * g value in the cell.
                    if (sigma_g > max_sigma_g) {
                        max_sigma_g = sigma_g;
                    }
                }
            }

            // Store the measured number of particles in the cell.
            number_particle_ptr[cell]    = static_cast<T>(count);

            // Store the maximum relative speed observed in the cell.
            max_relative_speed_ptr[cell] = max_relative;

            // If the cell cannot support a valid collision estimate, explicitly
            // set the collision count to zero and stop here.
            if (count < 2 || !(max_sigma_g > T(0))) {
                collision_count_ptr[cell] = 0;
                return;
            }

            // Compute the number of unique unordered particle pairs in the cell.
            const T pair_count = static_cast<T>(count) * static_cast<T>(count - 1) * T(0.5);

            // Estimate the number of NTC collision attempts for the cell.
            //
            // The estimate scales with:
            //   - the number of candidate pairs,
            //   - the maximum sigma * g value in the cell,
            //   - the statistical weight,
            //   - the time step,
            // and is normalized by the cell volume.
            const T ntc_count  = pair_count * max_sigma_g * statistical_weight * dt / cell_volume;

            // Convert the continuous estimate to a discrete number of collision
            // attempts by taking its floor.
            int collisions      = static_cast<int>(std::floor(ntc_count));

            // Compute the maximum number of distinct unordered pairs available
            // in the cell. Collision attempts are later clamped to this range.
            const int max_pairs = count * (count - 1) / 2;

            // Clamp the collision count defensively to a valid range.
            if (collisions < 0) {
                collisions = 0;
            } else if (collisions > max_pairs) {
                collisions = max_pairs;
            }

            // Store the final collision-attempt count for the cell.
            collision_count_ptr[cell] = collisions;
        });

    return true;
}

template <typename T>
bool
DsmcSolver<T>::build_flattened_collision_workload() noexcept {
    // Retrieve the per-cell collision-count state, which is the source data
    // used to construct the flattened collision-to-cell mapping.
    auto* collision_count_state
        = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    // Fail safely if the required state does not exist.
    if (collision_count_state == nullptr) {
        reset_collision_data();
        return false;
    }

    // Cache the collision-count buffer and its raw pointer representation.
    auto& collision_count = collision_count_state->data();
    auto* collision_count_ptr = atlas::raw_pointer_cast(collision_count.data());

    // Read the total number of cells in the universe.
    const int num_of_cells = this->_universe->number_of_cells();

    // Keep the offset buffer sized consistently with the cell count.
    if (_collision_offsets.size() != static_cast<std::size_t>(num_of_cells)) {
        _collision_offsets.resize(static_cast<std::size_t>(num_of_cells));
    }

    // Build an exclusive prefix sum over per-cell collision counts.
    //
    // The resulting offset of each cell points to the first index in the
    // flattened workload array that belongs to that cell.
    atlas::exclusive_scan<ExecutionPolicy::device>(
        collision_count.begin(),
        collision_count.end(),
        _collision_offsets.begin(),
        0);

    // Compute the total number of collision entries that the flattened array
    // must store.
    //
    // For an exclusive scan:
    //   total = last_offset + last_count
    const auto last_cell = static_cast<std::size_t>(num_of_cells - 1);
    const int total_collisions = _collision_offsets[last_cell] + collision_count[last_cell];

    // If no collisions exist globally, keep the flattened workload empty.
    if (total_collisions <= 0) {
        _flattened_collision_cells.resize(0);
        return false;
    }

    // Allocate one flattened entry per scheduled collision attempt.
    _flattened_collision_cells.resize(static_cast<std::size_t>(total_collisions));

    // Fill the flattened array so that each collision attempt stores the source
    // cell it belongs to.
    //
    // Example:
    //   collision_count[cell] = 3
    // then that cell index is written three times into the flattened array,
    // starting at its exclusive-scan offset.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=,
         collision_offsets_ptr = atlas::raw_pointer_cast(_collision_offsets.data()),
         flattened_collision_cells_ptr = atlas::raw_pointer_cast(_flattened_collision_cells.data())] ATLAS_DEVICE(
            const int cell) {
            const int collisions = collision_count_ptr[cell];

            // Cells with zero scheduled collisions contribute no entries.
            if (collisions <= 0) {
                return;
            }

            // Read the starting offset of the current cell in the flattened array.
            const int offset = collision_offsets_ptr[cell];

            // Write the cell index once for every local collision attempt.
            for (int local_collision = 0; local_collision < collisions; ++local_collision) {
                flattened_collision_cells_ptr[offset + local_collision] = cell;
            }
        });

    return true;
}

template <typename T>
int
DsmcSolver<T>::nth_valid_particle(const int nth,
                                  const int begin,
                                  const int end,
                                  const int particle_count,
                                  const int* indices_ptr) noexcept {
    // Convert the nth local particle ordinal within the cell range [begin, end)
    // into the corresponding index inside the sorted searcher array.
    const int sorted_index = begin + nth;

    // Reject invalid local ordinals or out-of-range accesses immediately.
    if (nth < 0 || sorted_index < begin || sorted_index >= end) {
        return -1;
    }

    // Resolve the global particle index stored at the computed sorted position.
    const int particle_index = indices_ptr[sorted_index];

    // Return the particle index only if it falls inside the valid global range.
    return (particle_index >= 0 && particle_index < particle_count) ? particle_index : -1;
}

template <typename T>
int
DsmcSolver<T>::pair_ordinal_to_rhs(const int count, int ordinal, int& lhs_local) noexcept {
    // Start from the first row of the conceptual upper-triangular pair matrix.
    //
    // For particles indexed locally as:
    //   0, 1, 2, ..., count - 1
    // the unordered pairs can be laid out row by row:
    //   row 0: (0,1), (0,2), ..., (0,count-1)
    //   row 1: (1,2), (1,3), ..., (1,count-1)
    //   ...
    //
    // This function maps a flat pair ordinal into:
    //   - lhs_local: the row index,
    //   - return value: the corresponding rhs index.
    lhs_local = 0;

    // Walk rows until the ordinal falls inside the current row width.
    while (lhs_local < count - 1) {
        // The number of pairs remaining in the current row decreases by one
        // for each increment of lhs_local.
        const int row_width = count - lhs_local - 1;

        // If the ordinal falls within the current row, return the matching rhs.
        if (ordinal < row_width) {
            return lhs_local + 1 + ordinal;
        }

        // Otherwise skip the entire current row and continue into the next row.
        ordinal -= row_width;
        ++lhs_local;
    }

    // Return -1 if the ordinal does not correspond to a valid pair.
    return -1;
}

template <typename T>
DsmcKernelType
DsmcSolver<T>::kernel_type() const noexcept {
    // Return the runtime-selected collision kernel type used by this solver.
    return _kernel_type;
}

template <typename T>
const DeviceBuffer<int>&
DsmcSolver<T>::collision_offsets() const noexcept {
    // Return the exclusive-scan offsets that map each cell to its start index
    // in the flattened collision workload.
    return _collision_offsets;
}

template <typename T>
const DeviceBuffer<int>&
DsmcSolver<T>::flattened_collision_cells() const noexcept {
    // Return the flattened array that stores one cell index per scheduled
    // collision attempt across the entire domain.
    return _flattened_collision_cells;
}

} // namespace atlas::system