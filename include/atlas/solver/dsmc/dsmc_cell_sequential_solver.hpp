#pragma once

namespace atlas::system {

template <typename T>
DsmcCellSequentialSolver<T>::DsmcCellSequentialSolver(UniverseHostPtr<T> universe,
                                                      FluidHostPtr<T> fluid,
                                                      SpatialHashingSearcherHostPtr<T> searcher,
                                                      const DsmcKernelType kernel_type) noexcept
    : DsmcSolver<T>(
        std::move(universe),
        std::move(fluid),
        std::move(searcher),
        kernel_type) { }

template <typename T>
typename DsmcCellSequentialSolver<T>::Builder
DsmcCellSequentialSolver<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
DsmcCellSequentialSolver<T>::apply_collision(const DeviceBuffer<int>* allocated_solver,
                                             const int index,
                                             const T) {
    // Use the DSMC runtime data prepared by the base solver.
    const auto probe = this->_probe;

    // Required buffers must exist before launching the collision kernel.
    if (probe.num_of_cells <= 0 || probe.collision_count_ptr == nullptr
        || probe.velocity_ptr == nullptr || probe.species_ptr == nullptr
        || probe.properties_ptr == nullptr
        || probe.indices_ptr == nullptr
        || probe.cell_start_ptr == nullptr || probe.cell_end_ptr == nullptr
        || probe.number_particle_ptr == nullptr || probe.max_sigma_g_ptr == nullptr) {
        return;
    }

    // Optional codec allocation restricts this solver to selected cells.
    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells assigned to another solver.
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int collisions = probe.collision_count_ptr[cell];
            const int count      = static_cast<int>(probe.number_particle_ptr[cell]);
            const T max_sigma_g  = probe.max_sigma_g_ptr[cell];

            // No collision trial is possible without pairs or a positive majorant.
            if (collisions <= 0 || count < 2 || !(max_sigma_g > T(0))) {
                return;
            }

            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            // Invalid or empty cell ranges are ignored.
            if (begin < 0 || end <= begin) {
                return;
            }

            // Process this cell's scheduled collision trials sequentially.
            for (int local_collision = 0; local_collision < collisions; ++local_collision) {
                // Derive an independent deterministic random stream per cell/trial.
                const auto stream = static_cast<std::uint64_t>(cell) * atlas::seed::DSMC_CELL_STREAM_MULTIPLIER
                    + static_cast<std::uint64_t>(local_collision);

                // Sample two distinct local particle indices inside the cell.
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
                const int particle_i = DsmcSolver<T>::nth_valid_particle(lhs_local,
                                                                         begin,
                                                                         end,
                                                                         probe.particle_count,
                                                                         probe.indices_ptr);
                const int particle_j = DsmcSolver<T>::nth_valid_particle(rhs_local,
                                                                         begin,
                                                                         end,
                                                                         probe.particle_count,
                                                                         probe.indices_ptr);

                if (particle_i < 0 || particle_j < 0) {
                    continue;
                }

                const std::size_t species_i = probe.species_ptr[particle_i];
                const std::size_t species_j = probe.species_ptr[particle_j];

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

                if (!(sigma_g > T(0))) {
                    continue;
                }

                // Accept with probability sigma_g / max_sigma_g.
                T accept_probability = sigma_g / max_sigma_g;
                if (accept_probability > T(1)) {
                    accept_probability = T(1);
                }

                const T accept_sample = atlas::sampling::sample_hashed_unit_interval<T>(
                    cell,
                    probe.collision_seed + stream + atlas::seed::DSMC_COLLISION_ACCEPT_SALT);

                if (accept_sample >= accept_probability) {
                    continue;
                }

                // Apply the selected DSMC collision kernel to accepted pairs.
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
typename DsmcCellSequentialSolver<T>::Builder&
DsmcCellSequentialSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename DsmcCellSequentialSolver<T>::Builder&
DsmcCellSequentialSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcCellSequentialSolver<T>::Builder&
DsmcCellSequentialSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename DsmcCellSequentialSolver<T>::Builder&
DsmcCellSequentialSolver<T>::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    _kernel = DsmcKernel<T>(kernel_type);
    return *this;
}

template <typename T>
void
DsmcCellSequentialSolver<T>::Builder::validate() const {
    // A complete DSMC runtime needs all three core dependencies.
    if (!_universe) {
        throw std::runtime_error("DsmcCellSequentialSolver::Builder: universe must not be null.");
    }

    if (!_fluid) {
        throw std::runtime_error("DsmcCellSequentialSolver::Builder: fluid must not be null.");
    }

    if (!_searcher) {
        throw std::runtime_error("DsmcCellSequentialSolver::Builder: searcher must not be null.");
    }
}

template <typename T>
DsmcCellSequentialSolver<T>
DsmcCellSequentialSolver<T>::Builder::build() const {
    validate();

    return DsmcCellSequentialSolver<T>(
        _universe,
        _fluid,
        _searcher,
        _kernel.type);
}

template <typename T>
atlas::host_shared_ptr<DsmcCellSequentialSolver<T>>
DsmcCellSequentialSolver<T>::Builder::make_host_shared() const {
    validate();

    return atlas::make_host_shared<DsmcCellSequentialSolver<T>>(
        _universe,
        _fluid,
        _searcher,
        _kernel.type);
}

} // namespace atlas::system