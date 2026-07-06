#include <atlas/solver/dsmc/dsmc_energy_exchange_solver.h>

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace atlas {

DsmcEnergyExchangeSolver::Builder
DsmcEnergyExchangeSolver::builder() noexcept {
    return Builder {};
}

void
DsmcEnergyExchangeSolver::apply_flattened_collision(const DeviceBuffer<int>* allocated_solver,
                                                    const int index) {
    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    if (!this->_flatten_workload.build(this->_probe.collision_count_ptr, this->_probe.cell_count, allocated_solver_ptr, index)) {
        return;
    }

    launch_flattened_energy_collisions();
}

void
DsmcEnergyExchangeSolver::launch_flattened_energy_collisions() {
    const auto probe                    = this->_probe;
    const int* collision_offsets_ptr    = atlas::raw_pointer_cast(this->_flatten_workload.collision_offsets.data());
    const int* collision_cells_ptr      = atlas::raw_pointer_cast(this->_flatten_workload.collision_cells.data());
    const int flattened_collision_count = this->_flatten_workload.flattened_collision_count;

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        flattened_collision_count,
        [=] ATLAS_ALL_DEVICE(const int work_index) {
            const int cell            = collision_cells_ptr[work_index];
            const int local_collision = work_index - collision_offsets_ptr[cell];
            const int count           = static_cast<int>(probe.number_particle_ptr[cell]);
            const float max_sigma_g   = probe.max_sigma_g_ptr[cell];
            const int begin           = probe.cell_start_ptr[cell];
            const int end             = probe.cell_end_ptr[cell];
            if (count < 2 || !(max_sigma_g > 0.0f) || begin < 0 || end <= begin) {
                return;
            }
            if (end - begin < count) {
                return;
            }

            const auto stream = static_cast<std::uint64_t>(cell) * atlas::DSMC_CELL_STREAM_MULTIPLIER
                + static_cast<std::uint64_t>(local_collision);
            int lhs_local = 0;
            int rhs_local = 0;
            DsmcSolver::sample_distinct_pair(
                lhs_local,
                rhs_local,
                cell,
                count,
                probe.collision_seed,
                stream);

            // Unlike DsmcSolver::collide_indexed_pair, local_collision is
            // passed through explicitly here: exchange_particle_internal_energy
            // needs it (alongside `stream`) to derive independent hash
            // salts per Larsen-Borgnakke mode draw within this collision.
            DsmcEnergyExchangeSolver::collide_indexed_pair(
                probe,
                cell,
                local_collision,
                stream,
                probe.indices_ptr[begin + lhs_local],
                probe.indices_ptr[begin + rhs_local],
                max_sigma_g);
        });
}

void
DsmcEnergyExchangeSolver::apply_collision(const DeviceBuffer<int>* allocated_solver,
                                          const int index,
                                          const float) {
    if (this->_workload_type == DsmcCollisionWorkloadType::flatten) {
        apply_flattened_collision(allocated_solver, index);
        return;
    }

    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    apply_cell_energy_collisions(allocated_solver_ptr, index);
}

void
DsmcEnergyExchangeSolver::apply_cell_energy_collisions(const int* allocated_solver_ptr, const int index) {
    const auto probe = this->_probe;

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        probe.cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            if (allocated_solver_ptr != nullptr && allocated_solver_ptr[cell] != index) {
                return;
            }

            const int collisions    = probe.collision_count_ptr[cell];
            const int count         = static_cast<int>(probe.number_particle_ptr[cell]);
            const float max_sigma_g = probe.max_sigma_g_ptr[cell];
            if (collisions <= 0 || count < 2 || !(max_sigma_g > 0.0f)) {
                return;
            }

            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];
            if (begin < 0 || end <= begin) {
                return;
            }
            if (end - begin < count) {
                return;
            }
            const auto stream_base = static_cast<std::uint64_t>(cell) * atlas::DSMC_CELL_STREAM_MULTIPLIER;

            for (int local_collision = 0; local_collision < collisions; ++local_collision) {
                const auto stream = stream_base + static_cast<std::uint64_t>(local_collision);
                int lhs_local     = 0;
                int rhs_local     = 0;
                DsmcSolver::sample_distinct_pair(
                    lhs_local,
                    rhs_local,
                    cell,
                    count,
                    probe.collision_seed,
                    stream);

                DsmcEnergyExchangeSolver::collide_indexed_pair(
                    probe,
                    cell,
                    local_collision,
                    stream,
                    probe.indices_ptr[begin + lhs_local],
                    probe.indices_ptr[begin + rhs_local],
                    max_sigma_g);
            }
        });
}

DsmcEnergyExchangeSolver::Builder&
DsmcEnergyExchangeSolver::Builder::with_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

DsmcEnergyExchangeSolver::Builder&
DsmcEnergyExchangeSolver::Builder::with_fluid(FluidHostPtr fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

DsmcEnergyExchangeSolver::Builder&
DsmcEnergyExchangeSolver::Builder::with_searcher(SearcherHostPtr searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

DsmcEnergyExchangeSolver::Builder&
DsmcEnergyExchangeSolver::Builder::with_kernel_type(const DsmcKernelType kernel_type) noexcept {
    _kernel_type = kernel_type;
    return *this;
}

DsmcEnergyExchangeSolver::Builder&
DsmcEnergyExchangeSolver::Builder::with_workload_type(const DsmcCollisionWorkloadType workload_type) noexcept {
    _workload_type = workload_type;
    return *this;
}

void
DsmcEnergyExchangeSolver::Builder::validate() const {
    if (!_universe) {
        throw std::runtime_error("DsmcEnergyExchangeSolver::Builder: universe must not be null.");
    }
    if (!_fluid) {
        throw std::runtime_error("DsmcEnergyExchangeSolver::Builder: fluid must not be null.");
    }
    if (!_searcher) {
        throw std::runtime_error("DsmcEnergyExchangeSolver::Builder: searcher must not be null.");
    }
}

DsmcEnergyExchangeSolver
DsmcEnergyExchangeSolver::Builder::build() const {
    validate();
    return DsmcEnergyExchangeSolver(_universe, _fluid, _searcher, _kernel_type, _workload_type);
}

atlas::host_shared_ptr<DsmcEnergyExchangeSolver>
DsmcEnergyExchangeSolver::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<DsmcEnergyExchangeSolver>(_universe, _fluid, _searcher, _kernel_type, _workload_type);
}

}
