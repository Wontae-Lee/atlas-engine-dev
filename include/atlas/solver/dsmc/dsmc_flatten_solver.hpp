#pragma once

#include <atlas/scan/exclusive_scan.h>

namespace atlas::system {

template <typename T>
DsmcFlattenSolver<T>::DsmcFlattenSolver(UniverseHostPtr<T> universe,
                                        FluidHostPtr<T> fluid,
                                        SpatialHashingSearcherHostPtr<T> searcher,
                                        const DsmcKernelType kernel_type,
                                        const bool pairing_without_replacement) noexcept
    : DsmcSolver<T>(
        std::move(universe),
        std::move(fluid),
        std::move(searcher),
        kernel_type,
        pairing_without_replacement) { }

template <typename T>
typename DsmcFlattenSolver<T>::Builder
DsmcFlattenSolver<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
bool
DsmcFlattenSolver<T>::build_flattened_collision_workload() {
    auto* collision_count_state = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    if (collision_count_state == nullptr) {
        this->reset_states();
        return false;
    }

    auto& collision_count     = collision_count_state->data();
    auto* collision_count_ptr = atlas::raw_pointer_cast(collision_count.data());
    const int num_of_cells    = this->_universe->number_of_cells();

    // One offset per cell is needed to map cell-local collisions into a flat range.
    if (_collision_offsets.size() != static_cast<std::size_t>(num_of_cells)) {
        _collision_offsets.resize(static_cast<std::size_t>(num_of_cells));
    }

    // Convert per-cell collision counts into prefix offsets.
    atlas::exclusive_scan<ExecutionPolicy::device>(
        collision_count.begin(),
        collision_count.end(),
        _collision_offsets.begin(),
        0);

    const auto last_cell  = static_cast<std::size_t>(num_of_cells - 1);
    const int last_offset = _collision_offsets[last_cell];
    const int last_count  = collision_count[last_cell];

    // Guard against overflowing the flattened work-item count.
    if (last_offset > std::numeric_limits<int>::max() - last_count) {
        throw std::overflow_error("DsmcFlattenSolver: flattened collision workload exceeds int range.");
    }

    const int total_collisions = last_offset + last_count;

    if (total_collisions <= 0) {
        _collision_cells.resize(0);
        _flattened_collision_count = 0;
        return false;
    }

    // Each flattened entry stores the cell that owns that collision trial.
    if (_collision_cells.size() != static_cast<std::size_t>(total_collisions)) {
        _collision_cells.resize(static_cast<std::size_t>(total_collisions));
    }

    _flattened_collision_count = total_collisions;

    auto* collision_offsets_ptr = atlas::raw_pointer_cast(_collision_offsets.data());
    auto* collision_cells_ptr   = atlas::raw_pointer_cast(_collision_cells.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int offset = collision_offsets_ptr[cell];
            const int count  = collision_count_ptr[cell];

            // Expand this cell's collision count into repeated cell ids.
            for (int local = 0; local < count; ++local) {
                collision_cells_ptr[offset + local] = cell;
            }
        });

    return true;
}

template <typename T>
void
DsmcFlattenSolver<T>::apply_collision(const DeviceBuffer<int>*,
                                      const int,
                                      const T) {
    // Build one flat work item for every scheduled collision trial.
    if (!build_flattened_collision_workload()) {
        return;
    }

    // Use the DSMC runtime data prepared by the base solver.
    const auto probe = this->_probe;

    const int* collision_offsets_ptr = atlas::raw_pointer_cast(_collision_offsets.data());
    const int* collision_cells_ptr   = atlas::raw_pointer_cast(_collision_cells.data());

    // Required data must exist before launching flattened collision work.
    if (_flattened_collision_count <= 0 || collision_offsets_ptr == nullptr || collision_cells_ptr == nullptr
        || probe.collision_count_ptr == nullptr || probe.properties_ptr == nullptr) {
        return;
    }

    if (probe.pairing_without_replacement) {
        apply_pairing_without_replacement_collision(probe, collision_offsets_ptr, collision_cells_ptr);
    } else {
        apply_random_pairing_collision(probe, collision_offsets_ptr, collision_cells_ptr);
    }
}

template <typename T>
void
DsmcFlattenSolver<T>::apply_random_pairing_collision(const Probe probe,
                                                     const int* collision_offsets_ptr,
                                                     const int* collision_cells_ptr) const {
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        _flattened_collision_count,
        [=] ATLAS_DEVICE(const int work_index) {
            // Resolve the owning cell and local collision id from the flat index.
            const int cell            = collision_cells_ptr[work_index];
            const int local_collision = work_index - collision_offsets_ptr[cell];

            const int count     = static_cast<int>(probe.number_particle_ptr[cell]);
            const T max_sigma_g = probe.max_sigma_g_ptr[cell];

            if (count < 2 || local_collision < 0 || !(max_sigma_g > T(0))) {
                return;
            }

            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            // Derive an independent deterministic random stream per cell/trial.
            const auto stream = static_cast<std::uint64_t>(cell) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
                + static_cast<std::uint64_t>(local_collision);

            // Sample two distinct local particle indices inside the owning cell.
            const int lhs_local = atlas::sampling::sample_hashed_index(
                cell,
                count,
                probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_LHS_SALT);

            int rhs_local = atlas::sampling::sample_hashed_index(
                cell,
                count - 1,
                probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_RHS_SALT);

            if (rhs_local >= lhs_local) {
                ++rhs_local;
            }

            // Convert local cell offsets to global particle indices.
            const int particle_i = DsmcSolver<T>::nth_valid_particle(
                lhs_local,
                begin,
                end,
                probe.particle_count,
                probe.indices_ptr);

            const int particle_j = DsmcSolver<T>::nth_valid_particle(
                rhs_local,
                begin,
                end,
                probe.particle_count,
                probe.indices_ptr);

            if (particle_i < 0 || particle_j < 0) {
                return;
            }

            const std::size_t species_i = probe.species_ptr[particle_i];
            const std::size_t species_j = probe.species_ptr[particle_j];

            const T accept_sample = atlas::sampling::sample_hashed_unit_interval<T>(
                cell,
                probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_ACCEPT_SALT);

            Vector3<T> lhs_velocity = probe.velocity_ptr[particle_i];
            Vector3<T> rhs_velocity = probe.velocity_ptr[particle_j];

            // Evaluate the actual sigma-g for the sampled pair.
            const T relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
            const T sigma_g                = DsmcSolver<T>::sigma_g(
                probe.kernel,
                probe.properties_ptr,
                species_i,
                species_j,
                relative_speed_squared);

            // Convert sigma-g into an acceptance probability.
            T accept_probability = T(0);
            if (sigma_g > T(0)) {
                accept_probability = sigma_g / max_sigma_g;
                if (accept_probability > T(1)) {
                    accept_probability = T(1);
                }
            }

            if (accept_sample < accept_probability) {
                // Apply the selected DSMC kernel to accepted pairs.
                probe.kernel(
                    lhs_velocity,
                    rhs_velocity,
                    probe.properties_ptr[species_i],
                    probe.properties_ptr[species_j]);

                // Commit updated velocities back to particle storage.
                probe.velocity_ptr[particle_i] = lhs_velocity;
                probe.velocity_ptr[particle_j] = rhs_velocity;
            }
        });
}

template <typename T>
void
DsmcFlattenSolver<T>::apply_pairing_without_replacement_collision(const Probe probe,
                                                     const int* collision_offsets_ptr,
                                                     const int* collision_cells_ptr) const {
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        _flattened_collision_count,
        [=] ATLAS_DEVICE(const int work_index) {
            const int cell            = collision_cells_ptr[work_index];
            const int local_collision = work_index - collision_offsets_ptr[cell];

            const int count     = static_cast<int>(probe.number_particle_ptr[cell]);
            const T max_sigma_g = probe.max_sigma_g_ptr[cell];

            if (count < 2 || local_collision < 0 || !(max_sigma_g > T(0))) {
                return;
            }

            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            const auto stream = static_cast<std::uint64_t>(cell) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
                + static_cast<std::uint64_t>(local_collision);

            int lhs_local = -1;
            int rhs_local = -1;
            DsmcSolver<T>::select_pair_offsets_without_replacement(
                lhs_local,
                rhs_local,
                local_collision,
                count,
                cell,
                probe.collision_seed + static_cast<std::uint64_t>(cell));

            const int particle_i = DsmcSolver<T>::nth_valid_particle(
                lhs_local,
                begin,
                end,
                probe.particle_count,
                probe.indices_ptr);
            const int particle_j = DsmcSolver<T>::nth_valid_particle(
                rhs_local,
                begin,
                end,
                probe.particle_count,
                probe.indices_ptr);

            if (particle_i < 0 || particle_j < 0) {
                return;
            }

            const std::size_t species_i = probe.species_ptr[particle_i];
            const std::size_t species_j = probe.species_ptr[particle_j];
            const T accept_sample = atlas::sampling::sample_hashed_unit_interval<T>(
                cell,
                probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_ACCEPT_SALT);

            Vector3<T> lhs_velocity = probe.velocity_ptr[particle_i];
            Vector3<T> rhs_velocity = probe.velocity_ptr[particle_j];
            const T relative_speed_squared = (lhs_velocity - rhs_velocity).length_squared();
            const T sigma_g = DsmcSolver<T>::sigma_g(
                probe.kernel,
                probe.properties_ptr,
                species_i,
                species_j,
                relative_speed_squared);

            T accept_probability = T(0);
            if (sigma_g > T(0)) {
                accept_probability = sigma_g / max_sigma_g;
                if (accept_probability > T(1)) {
                    accept_probability = T(1);
                }
            }

            if (accept_sample < accept_probability) {
                probe.kernel(
                    lhs_velocity,
                    rhs_velocity,
                    probe.properties_ptr[species_i],
                    probe.properties_ptr[species_j]);

                probe.velocity_ptr[particle_i] = lhs_velocity;
                probe.velocity_ptr[particle_j] = rhs_velocity;
            }
        });
}

template <typename T>
const DeviceBuffer<int>&
DsmcFlattenSolver<T>::collision_offsets() const noexcept {

    return _collision_offsets;
}

template <typename T>
void
DsmcFlattenSolver<T>::reset_states() {
    // Reset base DSMC statistics before clearing flattened workload buffers.
    DsmcSolver<T>::reset_states();

    _collision_offsets.resize(0);
    _collision_cells.resize(0);
    _flattened_collision_count = 0;
}

template <typename T>
typename DsmcFlattenSolver<T>::Builder&
DsmcFlattenSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename DsmcFlattenSolver<T>::Builder&
DsmcFlattenSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcFlattenSolver<T>::Builder&
DsmcFlattenSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename DsmcFlattenSolver<T>::Builder&
DsmcFlattenSolver<T>::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    _kernel = DsmcKernel<T>(kernel_type);
    return *this;
}

template <typename T>
typename DsmcFlattenSolver<T>::Builder&
DsmcFlattenSolver<T>::Builder::with_pairing_without_replacement(const bool enabled) noexcept {
    _pairing_without_replacement = enabled;
    return *this;
}

template <typename T>
void
DsmcFlattenSolver<T>::Builder::validate() const {
    // A valid DSMC solver needs all core runtime dependencies.
    if (!_universe) {
        throw std::runtime_error("DsmcFlattenSolver::Builder: universe must not be null.");
    }

    if (!_fluid) {
        throw std::runtime_error("DsmcFlattenSolver::Builder: fluid must not be null.");
    }

    if (!_searcher) {
        throw std::runtime_error("DsmcFlattenSolver::Builder: searcher must not be null.");
    }
}

template <typename T>
DsmcFlattenSolver<T>
DsmcFlattenSolver<T>::Builder::build() const {
    validate();

    return DsmcFlattenSolver<T>(
        _universe,
        _fluid,
        _searcher,
        _kernel.type,
        _pairing_without_replacement);
}

template <typename T>
atlas::host_shared_ptr<DsmcFlattenSolver<T>>
DsmcFlattenSolver<T>::Builder::make_host_shared() const {
    validate();

    return atlas::make_host_shared<DsmcFlattenSolver<T>>(
        _universe,
        _fluid,
        _searcher,
        _kernel.type,
        _pairing_without_replacement);
}

} // namespace atlas::system
