#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace atlas::system {

template <typename T>
DsmcSolver<T>::DsmcSolver(UniverseHostPtr<T> universe,
                          FluidHostPtr<T> fluid,
                          SpatialHashingSearcherHostPtr<T> searcher,
                          const DsmcKernelType kernel_type) noexcept
    : Solver<T>(std::move(universe), std::move(fluid), std::move(searcher))
    , _kernel(DsmcKernel<T>(kernel_type)) {
    // Make sure all per-cell DSMC states exist before the first solve step.
    ensure_states();
}

template <typename T>
void
DsmcSolver<T>::solve(const T dt) {

    // Process all cells when no external solver-allocation map is provided.
    solve(nullptr, 0, dt);
}

template <typename T>
void
DsmcSolver<T>::solve(const DeviceBuffer<int>* allocated_solver,
                     const int index,
                     const T dt) {

    // Ensure that the universe has the required DSMC state buffers.
    ensure_states();

    // Rebuild the spatial hash so cell particle ranges match the current positions.
    this->_searcher->build();

    // Cache all raw pointers and scalar constants needed by device-side kernels.
    make_probe();

    // Measure cell-local collision bounds and sample the number of collision trials.
    measure_cell_collision_statistics(allocated_solver, index, dt);

    // Apply stochastic DSMC pair collisions using the measured cell statistics.
    apply_collision(allocated_solver, index, dt);
}

template <typename T>
void
DsmcSolver<T>::ensure_states() {

    // All DSMC cell states must match the current number of universe cells.
    const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());

    // Create a state if it does not exist, or resize it if the cell count changed.
    const auto ensure_state = [this, number_of_cells]<typename State>() {
        if (auto* state = this->_universe->template state<State>();
            state == nullptr) {
            this->_universe->template emplace_state<State>(number_of_cells);
        } else if (state->data().size() != number_of_cells) {
            state->data().resize(number_of_cells);
        }
    };

    // Number of particles per cell.
    ensure_state.template operator()<UniverseNumberParticleState<T>>();

    // Maximum relative particle speed per cell.
    ensure_state.template operator()<UniverseMaxRelativeSpeedState<T>>();

    // Maximum sigma*g value per cell, used as the NTC acceptance upper bound.
    ensure_state.template operator()<UniverseMaxSigmaGState<T>>();

    // Number of candidate collision trials sampled for each cell.
    ensure_state.template operator()<UniverseCollisionCountState<int>>();
}

template <typename T>
void
DsmcSolver<T>::reset_states() {

    // State buffers may need to be created or resized before they can be reset.
    ensure_states();

    auto* number_particle_state    = this->_universe->template state<UniverseNumberParticleState<T>>();
    auto* max_relative_speed_state = this->_universe->template state<UniverseMaxRelativeSpeedState<T>>();
    auto* max_sigma_g_state        = this->_universe->template state<UniverseMaxSigmaGState<T>>();
    auto* collision_count_state    = this->_universe->template state<UniverseCollisionCountState<int>>();

    // Clear all per-cell collision statistics from the previous solve step.
    number_particle_state->reset();
    max_relative_speed_state->reset();
    max_sigma_g_state->reset();
    collision_count_state->reset();
}

template <typename T>
void
DsmcSolver<T>::make_probe() noexcept {

    // Fluid particle data used during pair selection and collision.
    _probe.velocity_ptr   = atlas::raw_pointer_cast(this->_fluid->template state<FluidVelocityState<T>>()->data().data());
    _probe.species_ptr    = atlas::raw_pointer_cast(this->_fluid->template state<FluidSpeciesState<T>>()->data().data());
    _probe.properties_ptr = atlas::raw_pointer_cast(this->_fluid->particle_properties().data());

    // Universe cell states written during the measurement pass and read during collision.
    _probe.number_particle_ptr    = atlas::raw_pointer_cast(this->_universe->template state<UniverseNumberParticleState<T>>()->data().data());
    _probe.max_relative_speed_ptr = atlas::raw_pointer_cast(this->_universe->template state<UniverseMaxRelativeSpeedState<T>>()->data().data());
    _probe.max_sigma_g_ptr        = atlas::raw_pointer_cast(this->_universe->template state<UniverseMaxSigmaGState<T>>()->data().data());
    _probe.collision_count_ptr    = atlas::raw_pointer_cast(this->_universe->template state<UniverseCollisionCountState<int>>()->data().data());

    // Spatial hashing arrays that define the sorted particle range of each cell.
    _probe.indices_ptr    = this->_searcher->indices();
    _probe.cell_start_ptr = this->_searcher->cell_start();
    _probe.cell_end_ptr   = this->_searcher->cell_end();

    // Scalar constants copied into the probe for device-side DSMC operations.
    _probe.particle_count     = static_cast<int>(this->_fluid->particle_count());
    _probe.num_of_cells       = this->_universe->number_of_cells();
    _probe.cell_volume        = this->_universe->cell_volume();
    _probe.statistical_weight = this->_fluid->statistical_weight();
    _probe.kernel             = _kernel;

    // Advance the seed once per probe construction to decorrelate solve steps.
    _probe.collision_seed = _collision_seed++;
}

template <typename T>
bool
DsmcSolver<T>::measure_cell_collision_statistics(const DeviceBuffer<int>* allocated_solver,
                                                 const int index,
                                                 const T dt) {

    // Copy the probe by value so the device lambda receives a stable snapshot.
    const auto probe = _probe;

    // Optional per-cell solver ownership map for hybrid or multi-solver workflows.
    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells assigned to a different solver.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            // Read the sorted particle range belonging to this cell.
            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];
            const int count = end - begin;

            T max_relative_squared = T(0);
            T max_sigma_g          = T(0);

            // Exhaustively scan all unordered particle pairs in the cell.
            for (int lhs_sorted_index = begin; lhs_sorted_index < end; ++lhs_sorted_index) {
                const int particle_i = probe.indices_ptr[lhs_sorted_index];

                for (int rhs_sorted_index = lhs_sorted_index + 1; rhs_sorted_index < end; ++rhs_sorted_index) {
                    const int particle_j        = probe.indices_ptr[rhs_sorted_index];
                    const std::size_t species_i = probe.species_ptr[particle_i];
                    const std::size_t species_j = probe.species_ptr[particle_j];

                    // Relative speed determines the DSMC collision-rate term.
                    const Vector3<T> relative_velocity = probe.velocity_ptr[particle_i] - probe.velocity_ptr[particle_j];
                    const T relative_speed_squared     = relative_velocity.length_squared();

                    if (relative_speed_squared > max_relative_squared) {
                        max_relative_squared = relative_speed_squared;
                    }

                    // sigma_g is the pair-specific collision cross-section term multiplied by relative speed.
                    const T sigma_g = probe.kernel.sigma_g(
                        probe.properties_ptr,
                        species_i,
                        species_j,
                        relative_speed_squared);

                    if (sigma_g > max_sigma_g) {
                        max_sigma_g = sigma_g;
                    }
                }
            }

            // Store the measured per-cell statistics.
            probe.number_particle_ptr[cell] = static_cast<T>(count);

            probe.max_relative_speed_ptr[cell] = max_relative_squared > T(0)
                ? static_cast<T>(std::sqrt(static_cast<double>(max_relative_squared)))
                : T(0);

            probe.max_sigma_g_ptr[cell] = max_sigma_g;

            // No collision is possible with fewer than two particles or a non-positive bound.
            if (count < 2 || !(max_sigma_g > T(0))) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            // Number of unordered candidate pairs in the cell: N * (N - 1) / 2.
            const T ntc_pair_count = static_cast<T>(count) * static_cast<T>(count - 1) * T(0.5);

            // Expected number of NTC collision trials for this cell and time step.
            const T ntc_count = ntc_pair_count
                * max_sigma_g
                * probe.statistical_weight
                * dt
                / probe.cell_volume;

            if (!(ntc_count > T(0))) {
                probe.collision_count_ptr[cell] = 0;
                return;
            }

            // Use stochastic rounding so the expected integer count matches ntc_count.
            const T base_count = std::floor(ntc_count);
            int collisions     = static_cast<int>(base_count);
            const T remainder  = ntc_count - base_count;

            // Add one extra collision with probability equal to the fractional remainder.
            if (remainder > T(0)
                && atlas::sampling::sample_hashed_unit_interval<T>(cell, probe.collision_seed) < remainder) {
                ++collisions;
            }

            probe.collision_count_ptr[cell] = collisions;
        });

    return true;
}

template <typename T>
DsmcKernelType
DsmcSolver<T>::kernel_type() const noexcept {

    // The kernel stores the selected DSMC collision model type.
    return _kernel.type;
}

template <typename T>
void
DsmcSolver<T>::apply_collision(const DeviceBuffer<int>* allocated_solver,
                               const int index,
                               const T) {

    // Copy the probe by value for safe device-lambda capture.
    const auto probe = _probe;

    // Optional per-cell solver ownership map.
    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells not handled by this solver instance.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            // Read the collision statistics measured in the previous pass.
            const int collisions = probe.collision_count_ptr[cell];
            const int count      = static_cast<int>(probe.number_particle_ptr[cell]);
            const T max_sigma_g  = probe.max_sigma_g_ptr[cell];

            // Nothing to do if there are no trials, too few particles, or no valid collision bound.
            if (collisions <= 0 || count < 2 || !(max_sigma_g > T(0))) {
                return;
            }

            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            for (int local_collision = 0; local_collision < collisions; ++local_collision) {
                // Build a deterministic random stream for this cell and collision attempt.
                const auto stream = static_cast<std::uint64_t>(cell) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
                    + static_cast<std::uint64_t>(local_collision);

                // Sample the first local particle index uniformly from [0, count).
                const int lhs_local = atlas::sampling::sample_hashed_index(
                    cell,
                    count,
                    probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_LHS_SALT);

                // Sample the second local particle index from [0, count - 1).
                const int rhs_sample = atlas::sampling::sample_hashed_index(
                    cell,
                    count - 1,
                    probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_RHS_SALT);

                int rhs_local = rhs_sample;

                // Shift the second index so it never equals lhs_local.
                if (rhs_local >= lhs_local) {
                    ++rhs_local;
                }

                // Attempt one accepted/rejected DSMC collision for the sampled pair.
                DsmcSolver<T>::collide_pair(
                    probe,
                    cell,
                    local_collision,
                    begin,
                    end,
                    lhs_local,
                    rhs_local,
                    max_sigma_g);
            }
        });
}

template <typename T>
bool
DsmcSolver<T>::collide_pair(const Probe& probe,
                            const int cell,
                            const int local_collision,
                            const int begin,
                            const int end,
                            const int lhs_local,
                            const int rhs_local,
                            const T max_sigma_g) noexcept {

    // Convert local cell indices into global particle indices.
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

    // Invalid sampled indices are rejected defensively.
    if (particle_i < 0 || particle_j < 0) {
        return false;
    }

    const std::size_t species_i = probe.species_ptr[particle_i];
    const std::size_t species_j = probe.species_ptr[particle_j];

    // Load velocities into local variables so the collision kernel can update them.
    Vector3<T> lhs_velocity = probe.velocity_ptr[particle_i];
    Vector3<T> rhs_velocity = probe.velocity_ptr[particle_j];

    // Evaluate the actual sigma*g value for this sampled pair.
    const T relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
    const T sigma_g = probe.kernel.sigma_g(
        probe.properties_ptr,
        species_i,
        species_j,
        relative_speed_squared);

    // A non-positive collision-rate term cannot produce a valid collision.
    if (!(sigma_g > T(0))) {
        return false;
    }

    // Accept with probability sigma_g / max_sigma_g, clamped to one for safety.
    T accept_probability = sigma_g / max_sigma_g;
    if (accept_probability > T(1)) {
        accept_probability = T(1);
    }

    // Reconstruct the deterministic stream used for this cell and local collision.
    const auto stream = static_cast<std::uint64_t>(cell) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
        + static_cast<std::uint64_t>(local_collision);

    // Sample the random value used for the acceptance-rejection test.
    const T accept_sample = atlas::sampling::sample_hashed_unit_interval<T>(
        cell,
        probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_ACCEPT_SALT);

    if (accept_sample >= accept_probability) {
        return false;
    }

    // Apply the selected collision model to the accepted pair.
    probe.kernel(
        lhs_velocity,
        rhs_velocity,
        probe.properties_ptr[species_i],
        probe.properties_ptr[species_j]);

    // Write the post-collision velocities back to the fluid state.
    probe.velocity_ptr[particle_i] = lhs_velocity;
    probe.velocity_ptr[particle_j] = rhs_velocity;

    return true;
}

template <typename T>
int
DsmcSolver<T>::particle_at(const int nth,
                           const int begin,
                           const int end,
                           const int particle_count,
                           const int* indices_ptr) noexcept {

    // Convert a local cell offset to the corresponding sorted particle-array index.
    const int sorted_index = begin + nth;

    // Reject invalid local offsets and out-of-range sorted indices.
    if (nth < 0 || sorted_index < begin || sorted_index >= end) {
        return -1;
    }

    // Resolve the global particle index from the searcher's sorted index buffer.
    const int particle_index = indices_ptr[sorted_index];

    // Validate the resolved global particle index before returning it.
    return (particle_index >= 0 && particle_index < particle_count) ? particle_index : -1;
}

}
