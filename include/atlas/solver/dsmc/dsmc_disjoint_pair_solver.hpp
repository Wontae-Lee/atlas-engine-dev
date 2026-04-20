#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>

namespace atlas::system {

template <typename T>
DsmcDisjointPairSolver<T>::DsmcDisjointPairSolver(UniverseHostPtr<T> universe,
                                                  FluidHostPtr<T> fluid,
                                                  SpatialHashingSearcherHostPtr<T> searcher,
                                                  const DsmcKernelType kernel_type) noexcept
    : DsmcSolver<T>(std::move(universe), std::move(fluid), std::move(searcher), kernel_type) { }

template <typename T>
typename DsmcDisjointPairSolver<T>::Builder
DsmcDisjointPairSolver<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
DsmcDisjointPairSolver<T>::apply_collisions(const DeviceBuffer<int>* allocated_solver, const int index, const T) {
    auto* velocity_state = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state  = this->_fluid->template state<atlas::fluid::FluidSpeciesState<T>>();
    auto* number_particle_state
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* collision_count_state
        = this->_universe->template state<atlas::universe::UniverseCollisionCountState<int>>();

    if (velocity_state == nullptr || species_state == nullptr || number_particle_state == nullptr
        || collision_count_state == nullptr) {
        return;
    }

    auto& velocities          = velocity_state->data();
    auto& particle_species    = species_state->data();
    auto& number_particle     = number_particle_state->data();
    auto& collision_count     = collision_count_state->data();
    auto& particle_properties = this->_fluid->particle_properties();

    auto* mutable_velocity_ptr      = atlas::raw_pointer_cast(velocities.data());
    const auto* species_ptr         = atlas::raw_pointer_cast(particle_species.data());
    const auto* number_particle_ptr = atlas::raw_pointer_cast(number_particle.data());
    const auto* collision_count_ptr = atlas::raw_pointer_cast(collision_count.data());
    const auto* properties_ptr      = atlas::raw_pointer_cast(particle_properties.data());
    const auto* indices_ptr         = this->_searcher->indices();
    const auto* cell_start_ptr      = this->_searcher->cell_start();
    const auto* cell_end_ptr        = this->_searcher->cell_end();
    const int particle_count        = static_cast<int>(this->_fluid->particle_count());
    const int num_of_properties     = static_cast<int>(particle_properties.size());
    const int num_of_cells          = this->_universe->number_of_cells();
    const auto kernel               = this->_kernel;
    const auto* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int count               = static_cast<int>(number_particle_ptr[cell]);
            const int disjoint_pair_count = count / 2;
            const int collisions          = collision_count_ptr[cell];
            if (disjoint_pair_count <= 0 || collisions <= 0) {
                return;
            }

            const int begin     = cell_start_ptr[cell];
            const int end       = cell_end_ptr[cell];
            const int collision_limit = collisions < disjoint_pair_count ? collisions : disjoint_pair_count;

            for (int local_collision = 0; local_collision < collision_limit; ++local_collision) {
                const int lhs_local = local_collision * 2;
                const int rhs_local = lhs_local + 1;

                const int particle_i = DsmcSolver<T>::nth_valid_particle(lhs_local, begin, end, particle_count, indices_ptr);
                const int particle_j = DsmcSolver<T>::nth_valid_particle(rhs_local, begin, end, particle_count, indices_ptr);
                if (particle_i < 0 || particle_j < 0) {
                    continue;
                }

                const std::size_t species_i = species_ptr[particle_i];
                const std::size_t species_j = species_ptr[particle_j];
                if (species_i >= static_cast<std::size_t>(num_of_properties)
                    || species_j >= static_cast<std::size_t>(num_of_properties)) {
                    continue;
                }

                Vector3<T> lhs_velocity = mutable_velocity_ptr[particle_i];
                Vector3<T> rhs_velocity = mutable_velocity_ptr[particle_j];

                kernel(lhs_velocity, rhs_velocity, properties_ptr[species_i], properties_ptr[species_j]);

                mutable_velocity_ptr[particle_i] = lhs_velocity;
                mutable_velocity_ptr[particle_j] = rhs_velocity;
            }
        });
}

template <typename T>
typename DsmcDisjointPairSolver<T>::Builder&
DsmcDisjointPairSolver<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename DsmcDisjointPairSolver<T>::Builder&
DsmcDisjointPairSolver<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename DsmcDisjointPairSolver<T>::Builder&
DsmcDisjointPairSolver<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename DsmcDisjointPairSolver<T>::Builder&
DsmcDisjointPairSolver<T>::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    _kernel_type = kernel_type;
    return *this;
}

template <typename T>
void
DsmcDisjointPairSolver<T>::Builder::validate() const {
    if (!_universe) {
        throw std::runtime_error("DsmcDisjointPairSolver::Builder: universe must not be null.");
    }

    if (!_fluid) {
        throw std::runtime_error("DsmcDisjointPairSolver::Builder: fluid must not be null.");
    }

    if (!_searcher) {
        throw std::runtime_error("DsmcDisjointPairSolver::Builder: searcher must not be null.");
    }

}

template <typename T>
DsmcDisjointPairSolver<T>
DsmcDisjointPairSolver<T>::Builder::build() const {
    validate();
    return DsmcDisjointPairSolver<T>(_universe, _fluid, _searcher, _kernel_type);
}

template <typename T>
atlas::host_shared_ptr<DsmcDisjointPairSolver<T>>
DsmcDisjointPairSolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<DsmcDisjointPairSolver<T>>(_universe, _fluid, _searcher, _kernel_type);
}

} // namespace atlas::system
