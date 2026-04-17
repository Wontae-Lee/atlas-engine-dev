#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>

namespace atlas::system {

template <typename T>
DsmcNtcSolver<T>::DsmcNtcSolver(UniverseHostPtr<T> universe,
                                FluidHostPtr<T> fluid,
                                SpatialHashingSearcherHostPtr<T> searcher,
                                const DsmcKernelType kernel_type,
                                const T collision_rate_scale) noexcept
    : DsmcSolver<T>(std::move(universe), std::move(fluid), std::move(searcher), kernel_type, collision_rate_scale) { }

template <typename T>
typename DsmcNtcSolver<T>::Builder
DsmcNtcSolver<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
DsmcNtcSolver<T>::apply_collisions(const DeviceBuffer<int>* allocated_solver, const int index, const T) {
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();

    if (velocity_state == nullptr || species_state == nullptr || number_particle_state == nullptr) {
        return;
    }

    auto& velocities          = velocity_state->data();
    auto& particle_species    = species_state->data();
    auto& number_particle     = number_particle_state->data();
    auto& particle_properties = this->_fluid->particle_properties();

    auto* mutable_velocity_ptr                = atlas::raw_pointer_cast(velocities.data());
    const auto* species_ptr                   = atlas::raw_pointer_cast(particle_species.data());
    const auto* number_particle_ptr           = atlas::raw_pointer_cast(number_particle.data());
    const auto* properties_ptr                = atlas::raw_pointer_cast(particle_properties.data());
    const auto* indices_ptr                   = this->_searcher->indices();
    const auto* cell_start_ptr                = this->_searcher->cell_start();
    const auto* cell_end_ptr                  = this->_searcher->cell_end();
    const int particle_count                  = static_cast<int>(this->_fluid->particle_count());
    const int num_of_properties               = static_cast<int>(particle_properties.size());
    const int total_collisions                = static_cast<int>(this->_flattened_collision_cells.size());
    const auto kernel                         = this->_kernel;
    const auto* allocated_solver_ptr          = allocated_solver != nullptr ? atlas::raw_pointer_cast(allocated_solver->data()) : nullptr;
    const auto* collision_offsets_ptr         = atlas::raw_pointer_cast(this->_collision_offsets.data());
    const auto* flattened_collision_cells_ptr = atlas::raw_pointer_cast(this->_flattened_collision_cells.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        total_collisions,
        [=] ATLAS_DEVICE(const int global_collision) {
            const int cell = flattened_collision_cells_ptr[global_collision];
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int count = static_cast<int>(number_particle_ptr[cell]);
            if (count < 2) {
                return;
            }

            const int begin           = cell_start_ptr[cell];
            const int end             = cell_end_ptr[cell];
            const int pair_count      = count * (count - 1) / 2;
            const int local_collision = global_collision - collision_offsets_ptr[cell];
            int lhs_local             = 0;
            const int ordinal         = local_collision % pair_count;
            const int rhs_local       = DsmcSolver<T>::pair_ordinal_to_rhs(count, ordinal, lhs_local);

            const int particle_i = DsmcSolver<T>::nth_valid_particle(lhs_local, begin, end, particle_count, indices_ptr);
            const int particle_j = DsmcSolver<T>::nth_valid_particle(rhs_local, begin, end, particle_count, indices_ptr);
            if (particle_i < 0 || particle_j < 0) {
                return;
            }

            const std::size_t species_i = species_ptr[particle_i];
            const std::size_t species_j = species_ptr[particle_j];
            if (species_i >= static_cast<std::size_t>(num_of_properties)
                || species_j >= static_cast<std::size_t>(num_of_properties)) {
                return;
            }

            Vector3<T> lhs_velocity = mutable_velocity_ptr[particle_i];
            Vector3<T> rhs_velocity = mutable_velocity_ptr[particle_j];

            kernel(lhs_velocity, rhs_velocity, properties_ptr[species_i], properties_ptr[species_j]);

            mutable_velocity_ptr[particle_i] = lhs_velocity;
            mutable_velocity_ptr[particle_j] = rhs_velocity;
        });
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
typename DsmcNtcSolver<T>::Builder&
DsmcNtcSolver<T>::Builder::with_collision_rate_scale(const T collision_rate_scale) noexcept {
    _collision_rate_scale = collision_rate_scale;
    return *this;
}

template <typename T>
void
DsmcNtcSolver<T>::Builder::validate() const {
    if (!_universe) {
        throw std::runtime_error("DsmcNtcSolver::Builder: universe must not be null.");
    }

    if (!_fluid) {
        throw std::runtime_error("DsmcNtcSolver::Builder: fluid must not be null.");
    }

    if (!_searcher) {
        throw std::runtime_error("DsmcNtcSolver::Builder: searcher must not be null.");
    }

    if (!(_collision_rate_scale > T(0))) {
        throw std::runtime_error("DsmcNtcSolver::Builder: collision_rate_scale must be positive.");
    }
}

template <typename T>
DsmcNtcSolver<T>
DsmcNtcSolver<T>::Builder::build() const {
    validate();
    return DsmcNtcSolver<T>(_universe, _fluid, _searcher, _kernel_type, _collision_rate_scale);
}

template <typename T>
atlas::host_shared_ptr<DsmcNtcSolver<T>>
DsmcNtcSolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<DsmcNtcSolver<T>>(_universe, _fluid, _searcher, _kernel_type, _collision_rate_scale);
}

} // namespace atlas::system
