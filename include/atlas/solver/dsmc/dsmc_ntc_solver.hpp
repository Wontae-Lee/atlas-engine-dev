#pragma once

#include <atlas/parallel/parallel_for.h>

#include <stdexcept>

namespace atlas::system {

template <typename T>
DsmcNtcSolver<T>::DsmcNtcSolver(UniverseHostPtr<T> universe,
                                FluidHostPtr<T> fluid,
                                SpatialHashingSearcherHostPtr<T> searcher,
                                const DsmcKernelType kernel_type) noexcept
    : DsmcSolver<T>(std::move(universe), std::move(fluid), std::move(searcher), kernel_type) {
}

template <typename T>
typename DsmcNtcSolver<T>::Builder
DsmcNtcSolver<T>::builder() noexcept {
    // Return a fresh builder for fluent solver construction.
    return Builder {};
}

template <typename T>
void
DsmcNtcSolver<T>::apply_collisions(const typename DsmcSolver<T>::DsmcSolverProbe& probe,
                                   const int index,
                                   const T) {
    // Process each cell independently on the device.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Skip cells assigned to a different solver in hybrid-solver mode.
            if (probe.allocated_solver_ptr != nullptr && probe.allocated_solver_ptr[cell] != index) {
                return;
            }

            // Count all possible unique unordered pairs in this cell.
            const int count      = static_cast<int>(probe.number_particle_ptr[cell]);
            const int pair_count = count * (count - 1) / 2;

            // Number of collision attempts requested for this cell.
            const int collisions = probe.collision_count_ptr[cell];

            // At least two particles and one requested collision are required.
            if (count < 2 || pair_count <= 0 || collisions <= 0) {
                return;
            }

            // Get the sorted-particle range for this cell.
            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            for (int local_collision = 0; local_collision < collisions; ++local_collision) {
                // Wrap collision attempts over all possible local unordered pairs.
                int lhs_local     = 0;
                const int ordinal = local_collision % pair_count;

                // Convert pair ordinal to local pair indices.
                const int rhs_local = DsmcSolver<T>::pair_ordinal_to_rhs(count, ordinal, lhs_local);

                // Convert local valid-particle positions into global particle indices.
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

                // Skip incomplete or invalid pairs.
                if (particle_i < 0 || particle_j < 0) {
                    continue;
                }

                // Resolve material/species data for both particles.
                const std::size_t species_i = probe.species_ptr[particle_i];
                const std::size_t species_j = probe.species_ptr[particle_j];

                if (species_i >= static_cast<std::size_t>(probe.num_of_properties)
                    || species_j >= static_cast<std::size_t>(probe.num_of_properties)) {
                    continue;
                }

                // Copy velocities locally so the collision kernel can update the pair.
                Vector3<T> lhs_velocity = probe.velocity_ptr[particle_i];
                Vector3<T> rhs_velocity = probe.velocity_ptr[particle_j];

                // Apply the configured DSMC collision model.
                probe.kernel(
                    lhs_velocity,
                    rhs_velocity,
                    probe.properties_ptr[species_i],
                    probe.properties_ptr[species_j]);

                // Write the post-collision velocities back to the fluid state.
                probe.velocity_ptr[particle_i] = lhs_velocity;
                probe.velocity_ptr[particle_j] = rhs_velocity;
            }
        });
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    // Store universe dependency.
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    // Store fluid dependency.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    // Store searcher dependency.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    // Store selected DSMC collision kernel type.
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
void
DsmcNtcSolver<T>::Builder::validate() const {
    // Builder requires all runtime dependencies.
    if (!_universe) {
        throw std::runtime_error("DsmcNtcSolver::Builder: universe must not be null.");
    }

    if (!_fluid) {
        throw std::runtime_error("DsmcNtcSolver::Builder: fluid must not be null.");
    }

    if (!_searcher) {
        throw std::runtime_error("DsmcNtcSolver::Builder: searcher must not be null.");
    }
}

template <typename T>
DsmcNtcSolver<T>
DsmcNtcSolver<T>::Builder::build() const {
    // Validate configuration and construct a solver value.
    validate();
    return DsmcNtcSolver<T>(_universe, _fluid, _searcher, _kernel_type);
}

template <typename T>
atlas::host_shared_ptr<DsmcNtcSolver<T>>
DsmcNtcSolver<T>::Builder::make_host_shared() const {
    // Validate configuration and construct a host-shared solver.
    validate();
    return atlas::make_host_shared<DsmcNtcSolver<T>>(_universe, _fluid, _searcher, _kernel_type);
}

} // namespace atlas::system