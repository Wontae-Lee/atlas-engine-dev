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
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher))
    , _kernel(DsmcKernel<T>(kernel_type))
    , _kernel_type(kernel_type) {
    // Ensure per-cell DSMC statistic states exist when a universe is attached.
    ensure_universe_states();
}

template <typename T>
void
DsmcSolver<T>::solve(const T dt) {
    // Run DSMC without solver-index filtering.
    solve(nullptr, 0, dt);
}

template <typename T>
void
DsmcSolver<T>::solve(const DeviceBuffer<int>* allocated_solver,
                     const int index,
                     const T dt) {
    // Required runtime dependencies must exist before collision work can be built.
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        reset_collision_data();
        return;
    }

    // Prepare universe states and rebuild cell-local particle ranges.
    ensure_universe_states();
    this->_searcher->build();

    // Collision-count estimation requires a positive time step.
    if (!(dt > T(0))) {
        throw std::invalid_argument("DsmcSolver: dt must be positive.");
    }

    // Build a raw-pointer view over all DSMC runtime data.
    DsmcSolverProbe probe;
    if (!make_probe(allocated_solver, probe)) {
        reset_collision_data();
        return;
    }

    // Measure per-cell collision statistics before applying the derived strategy.
    if (!build_collision_workload(probe, index, dt)) {
        return;
    }

    // Delegate concrete pair selection and velocity updates to the derived solver.
    apply_collisions(probe, index, dt);
}

template <typename T>
void
DsmcSolver<T>::ensure_universe_states() {
    // Nothing can be initialized without a universe.
    if (!this->_universe) {
        return;
    }

    const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());

    // Store the measured number of particles per cell.
    if (!this->_universe->template has_state<atlas::universe::UniverseNumberParticleState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseNumberParticleState<T>>(
            number_of_cells);
    }

    // Store the maximum relative speed observed in each cell.
    if (!this->_universe->template has_state<atlas::universe::UniverseMaxRelativeSpeedState<T>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseMaxRelativeSpeedState<T>>(
            number_of_cells);
    }

    // Store the number of collision attempts scheduled for each cell.
    if (!this->_universe->template has_state<atlas::universe::UniverseCollisionCountState<int>>()) {
        this->_universe->template emplace_state<atlas::universe::UniverseCollisionCountState<int>>(
            number_of_cells);
    }

    // Keep one offset entry per cell for optional flattened workload construction.
    _collision_offsets.resize(number_of_cells);
}

template <typename T>
void
DsmcSolver<T>::reset_collision_data() {
    // Without a universe, only solver-owned workload buffers can be cleared.
    if (!this->_universe) {
        _collision_offsets.resize(0);
        _flattened_collision_cells.resize(0);
        return;
    }

    const auto num_of_cells = this->_universe->number_of_cells();

    // Empty universe means no per-cell collision work is possible.
    if (num_of_cells <= 0) {
        _collision_offsets.resize(0);
        _flattened_collision_cells.resize(0);
        return;
    }

    // Ensure reset targets exist.
    ensure_universe_states();

    auto* number_particle_state    = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();
    auto* collision_count_state    = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    // Reset measured cell particle counts.
    if (number_particle_state != nullptr) {
        auto& buffer = number_particle_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), T(0));
    }

    // Reset measured maximum relative speeds.
    if (max_relative_speed_state != nullptr) {
        auto& buffer = max_relative_speed_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), T(0));
    }

    // Reset scheduled collision counts.
    if (collision_count_state != nullptr) {
        auto& buffer = collision_count_state->data();
        atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), 0);
    }

    // Reset flattened-workload offsets.
    if (_collision_offsets.size() != static_cast<std::size_t>(num_of_cells)) {
        _collision_offsets.resize(static_cast<std::size_t>(num_of_cells));
    }

    atlas::parallel_fill<ExecutionPolicy::device>(
        _collision_offsets.begin(),
        _collision_offsets.end(),
        0);

    // Clear the optional flattened cell list.
    _flattened_collision_cells.resize(0);
}

template <typename T>
bool
DsmcSolver<T>::build_collision_workload(const DsmcSolverProbe& probe,
                                        const int index,
                                        const T dt) {
    // Workload estimation depends directly on dt.
    if (!(dt > T(0))) {
        throw std::invalid_argument("DsmcSolver: dt must be positive.");
    }

    // Current implementation builds only per-cell collision statistics.
    return measure_cell_collision_statistics(probe, index, dt);
}

template <typename T>
bool
DsmcSolver<T>::initialize_collision_context() noexcept {
    // Required dependencies must exist before any DSMC context is usable.
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        reset_collision_data();
        return false;
    }

    // Ensure states exist and rebuild searcher ranges for the current particles.
    ensure_universe_states();
    this->_searcher->build();

    // Required particle states.
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    // Required universe collision-statistic states.
    auto* number_particle_state    = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();
    auto* collision_count_state    = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    if (velocity_state == nullptr || species_state == nullptr || number_particle_state == nullptr
        || max_relative_speed_state == nullptr || collision_count_state == nullptr) {
        reset_collision_data();
        return false;
    }

    return true;
}

template <typename T>
bool
DsmcSolver<T>::make_probe(const DeviceBuffer<int>* allocated_solver,
                          DsmcSolverProbe& probe) noexcept {
    // Probe construction requires all runtime dependencies.
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        return false;
    }

    // Retrieve required particle states.
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();

    // Retrieve required universe collision-statistic states.
    auto* number_particle_state    = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state = this->_universe->template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>();
    auto* collision_count_state    = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    if (velocity_state == nullptr || species_state == nullptr || number_particle_state == nullptr
        || max_relative_speed_state == nullptr || collision_count_state == nullptr) {
        return false;
    }

    // Extract buffers and species material properties.
    auto& velocities         = velocity_state->data();
    auto& species            = species_state->data();
    auto& number_particle    = number_particle_state->data();
    auto& max_relative_speed = max_relative_speed_state->data();
    auto& collision_count    = collision_count_state->data();
    auto& properties         = this->_fluid->particle_properties();

    // Store fluid pointers.
    probe.velocity_ptr   = atlas::raw_pointer_cast(velocities.data());
    probe.species_ptr    = atlas::raw_pointer_cast(species.data());
    probe.properties_ptr = atlas::raw_pointer_cast(properties.data());

    // Store universe statistic pointers.
    probe.number_particle_ptr    = atlas::raw_pointer_cast(number_particle.data());
    probe.max_relative_speed_ptr = atlas::raw_pointer_cast(max_relative_speed.data());
    probe.collision_count_ptr    = atlas::raw_pointer_cast(collision_count.data());

    // Store searcher cell-local particle ranges.
    probe.indices_ptr    = this->_searcher->indices();
    probe.cell_start_ptr = this->_searcher->cell_start();
    probe.cell_end_ptr   = this->_searcher->cell_end();

    // Optional solver allocation pointer for hybrid/coded cell filtering.
    probe.allocated_solver_ptr = allocated_solver != nullptr ? atlas::raw_pointer_cast(allocated_solver->data()) : nullptr;

    // Store scalar metadata required by DSMC device kernels.
    probe.particle_count     = static_cast<int>(this->_fluid->particle_count());
    probe.num_of_cells       = this->_universe->number_of_cells();
    probe.num_of_properties  = static_cast<int>(properties.size());
    probe.cell_volume        = this->_universe->cell_volume();
    probe.statistical_weight = this->_fluid->statistical_weight();
    probe.kernel_type        = _kernel_type;
    probe.kernel             = _kernel;

    return true;
}

template <typename T>
bool
DsmcSolver<T>::measure_cell_collision_statistics(const DsmcSolverProbe& probe,
                                                 const int index,
                                                 const T dt) {
    // Basic DSMC preconditions for collision-count estimation.
    if (probe.particle_count < 2 || probe.num_of_cells <= 0 || probe.num_of_properties <= 0
        || !(probe.cell_volume > T(0))) {
        reset_collision_data();
        return false;
    }

    // Compute per-cell particle count, maximum relative speed, and collision count.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Exclude cells assigned to a different solver.
            if (probe.allocated_solver_ptr != nullptr && probe.allocated_solver_ptr[cell] != index) {
                probe.number_particle_ptr[cell]    = T(0);
                probe.max_relative_speed_ptr[cell] = T(0);
                probe.collision_count_ptr[cell]    = 0;
                return;
            }

            // Get the sorted-particle range for this cell.
            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (begin < 0 || end <= begin) {
                probe.number_particle_ptr[cell]    = T(0);
                probe.max_relative_speed_ptr[cell] = T(0);
                probe.collision_count_ptr[cell]    = 0;
                return;
            }

            // Count particles directly from the searcher range.
            const int count = end - begin;
            T max_relative  = T(0);
            T max_sigma_g   = T(0);

            // Scan all unique unordered particle pairs in this cell.
            for (int a = begin; a < end; ++a) {
                const int particle_i = probe.indices_ptr[a];

                for (int b = a + 1; b < end; ++b) {
                    const int particle_j = probe.indices_ptr[b];

                    const std::size_t species_i = probe.species_ptr[particle_i];
                    const std::size_t species_j = probe.species_ptr[particle_j];

                    // Ignore pairs with invalid material/species references.
                    if (species_i >= static_cast<std::size_t>(probe.num_of_properties)
                        || species_j >= static_cast<std::size_t>(probe.num_of_properties)) {
                        continue;
                    }

                    // Track relative speed and sigma*g maximum for NTC estimation.
                    const Vector3<T> relative_velocity = probe.velocity_ptr[particle_i] - probe.velocity_ptr[particle_j];
                    const T relative_speed             = relative_velocity.length();

                    if (relative_speed > max_relative) {
                        max_relative = relative_speed;
                    }

                    const T sigma = DsmcKernel<T>::cross_section(
                        probe.kernel_type,
                        probe.properties_ptr[species_i],
                        probe.properties_ptr[species_j],
                        relative_speed);

                    const T sigma_g = sigma * relative_speed;

                    if (sigma_g > max_sigma_g) {
                        max_sigma_g = sigma_g;
                    }
                }
            }

            // Store measured cell statistics.
            probe.number_particle_ptr[cell]    = static_cast<T>(count);
            probe.max_relative_speed_ptr[cell] = max_relative;

            // Cells without a valid collision rate receive no scheduled collisions.
            if (count < 2 || !(max_sigma_g > T(0))) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            // Estimate NTC collision attempts for this cell.
            const T pair_count = static_cast<T>(count) * static_cast<T>(count - 1) * T(0.5);
            const T ntc_count  = pair_count * max_sigma_g * probe.statistical_weight * dt / probe.cell_volume;

            int collisions      = static_cast<int>(std::floor(ntc_count));
            const int max_pairs = count * (count - 1) / 2;

            // Clamp the scheduled count to the available unordered pairs.
            if (collisions < 0) {
                collisions = 0;
            } else if (collisions > max_pairs) {
                collisions = max_pairs;
            }

            probe.collision_count_ptr[cell] = collisions;
        });

    return true;
}

template <typename T>
bool
DsmcSolver<T>::build_flattened_collision_workload() noexcept {
    auto* collision_count_state = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    // Cannot build a flattened workload without collision counts.
    if (collision_count_state == nullptr) {
        reset_collision_data();
        return false;
    }

    auto& collision_count     = collision_count_state->data();
    auto* collision_count_ptr = atlas::raw_pointer_cast(collision_count.data());
    const int num_of_cells    = this->_universe->number_of_cells();

    // Ensure offset buffer has one entry per cell.
    if (_collision_offsets.size() != static_cast<std::size_t>(num_of_cells)) {
        _collision_offsets.resize(static_cast<std::size_t>(num_of_cells));
    }

    // Compute exclusive offsets from per-cell collision counts.
    atlas::exclusive_scan<ExecutionPolicy::device>(
        collision_count.begin(),
        collision_count.end(),
        _collision_offsets.begin(),
        0);

    // Last offset plus last count gives the total number of flattened entries.
    const auto last_cell       = static_cast<std::size_t>(num_of_cells - 1);
    const int total_collisions = _collision_offsets[last_cell] + collision_count[last_cell];

    if (total_collisions <= 0) {
        _flattened_collision_cells.resize(0);
        return false;
    }

    _flattened_collision_cells.resize(static_cast<std::size_t>(total_collisions));

    // Fill each flattened slot with the cell that owns that scheduled collision.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [                              =,
         collision_offsets_ptr         = atlas::raw_pointer_cast(_collision_offsets.data()),
         flattened_collision_cells_ptr = atlas::raw_pointer_cast(_flattened_collision_cells.data())] ATLAS_DEVICE(const int cell) {
            const int collisions = collision_count_ptr[cell];

            if (collisions <= 0) {
                return;
            }

            const int offset = collision_offsets_ptr[cell];

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
    // Convert local offset to sorted-index position.
    const int sorted_index = begin + nth;

    // Reject out-of-range local offsets.
    if (nth < 0 || sorted_index < begin || sorted_index >= end) {
        return -1;
    }

    // Return the particle index only if it is globally valid.
    const int particle_index = indices_ptr[sorted_index];
    return (particle_index >= 0 && particle_index < particle_count) ? particle_index : -1;
}

template <typename T>
int
DsmcSolver<T>::pair_ordinal_to_rhs(const int count,
                                   int ordinal,
                                   int& lhs_local) noexcept {
    // Enumerate unordered local pairs row by row:
    // (0,1), (0,2), ..., (1,2), (1,3), ...
    lhs_local = 0;

    while (lhs_local < count - 1) {
        const int row_width = count - lhs_local - 1;

        if (ordinal < row_width) {
            return lhs_local + 1 + ordinal;
        }

        ordinal -= row_width;
        ++lhs_local;
    }

    return -1;
}

template <typename T>
DsmcKernelType
DsmcSolver<T>::kernel_type() const noexcept {
    // Return the selected DSMC kernel type.
    return _kernel_type;
}

template <typename T>
const DeviceBuffer<int>&
DsmcSolver<T>::collision_offsets() const noexcept {
    // Return per-cell offsets for the optional flattened workload.
    return _collision_offsets;
}

template <typename T>
const DeviceBuffer<int>&
DsmcSolver<T>::flattened_collision_cells() const noexcept {
    // Return the optional flattened collision-cell list.
    return _flattened_collision_cells;
}

} // namespace atlas::system