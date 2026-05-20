#pragma once

#include <atlas/scan/exclusive_scan.h>

namespace atlas::system {

template <typename T>
DsmcFlattenSolver<T>::DsmcFlattenSolver(UniverseHostPtr<T> universe,
                                        FluidHostPtr<T> fluid,
                                        SpatialHashingSearcherHostPtr<T> searcher,
                                        const DsmcKernelType kernel_type) noexcept
    : DsmcSolver<T>(
        std::move(universe),
        std::move(fluid),
        std::move(searcher),
        kernel_type) { }

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

    if (_collision_offsets.size() != static_cast<std::size_t>(num_of_cells)) {
        _collision_offsets.resize(static_cast<std::size_t>(num_of_cells));
    }

    atlas::exclusive_scan<ExecutionPolicy::device>(
        collision_count.begin(),
        collision_count.end(),
        _collision_offsets.begin(),
        0);

    const auto last_cell  = static_cast<std::size_t>(num_of_cells - 1);
    const int last_offset = _collision_offsets[last_cell];
    const int last_count  = collision_count[last_cell];

    if (last_offset > std::numeric_limits<int>::max() - last_count) {
        throw std::overflow_error("DsmcFlattenSolver: flattened collision workload exceeds int range.");
    }

    const int total_collisions = last_offset + last_count;

    if (total_collisions <= 0) {
        _collision_cells.resize(0);
        _flattened_collision_count = 0;
        return false;
    }

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
    if (!build_flattened_collision_workload()) {
        return;
    }

    const auto probe = this->_probe;

    const int* collision_offsets_ptr = atlas::raw_pointer_cast(_collision_offsets.data());
    const int* collision_cells_ptr   = atlas::raw_pointer_cast(_collision_cells.data());

    if (_flattened_collision_count <= 0 || collision_offsets_ptr == nullptr || collision_cells_ptr == nullptr
        || probe.collision_count_ptr == nullptr || probe.properties_ptr == nullptr) {
        return;
    }

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
            const T sigma_g                = DsmcSolver<T>::sigma_g(
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
void
DsmcFlattenSolver<T>::Builder::validate() const {
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
        _kernel.type);
}

template <typename T>
atlas::host_shared_ptr<DsmcFlattenSolver<T>>
DsmcFlattenSolver<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<DsmcFlattenSolver<T>>(
        _universe,
        _fluid,
        _searcher,
        _kernel.type);
}

}
