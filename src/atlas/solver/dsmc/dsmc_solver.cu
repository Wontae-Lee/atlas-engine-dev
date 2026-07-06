#include <atlas/solver/dsmc/dsmc_solver.h>

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/solver/dsmc/detail/dsmc_probe_builder.h>
#include <atlas/universe/universe_state.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace atlas {

DsmcSolver::DsmcSolver(UniverseHostPtr universe,
                       FluidHostPtr fluid,
                       SearcherHostPtr searcher,
                       const DsmcKernelType kernel_type,
                       const DsmcCollisionWorkloadType workload_type) noexcept
    : Solver(std::move(universe), std::move(fluid), std::move(searcher))
    , _kernel(DsmcKernel(kernel_type))
    , _workload_type(workload_type) {

    ensure_states();
}

void
DsmcSolver::solve(const float dt) {

    solve(nullptr, 0, dt);
}

void
DsmcSolver::solve(const DeviceBuffer<int>* allocated_solver,
                  const int index,
                  const float dt) {

    ensure_states();

    // The searcher must be (re)built before make_probe() captures its
    // indices/cell_start/cell_end arrays, so the probe reflects this
    // step's particle positions rather than a stale partition.
    this->_searcher->build();

    make_probe();

    // Order matters: statistics must be measured (candidate counts +
    // max_sigma_g estimated) against this step's probe before
    // apply_collision draws and tests candidates against those counts.
    measure_collision_statistics(allocated_solver, index, dt);

    apply_collision(allocated_solver, index, dt);
}

void
DsmcSolver::ensure_states() {
    if (!this->_universe) {
        return;
    }

    const auto count = static_cast<std::size_t>(this->_universe->cell_count());

    if (auto* state = this->_universe->state<UniverseNumberParticleState>();
        state == nullptr) {
        this->_universe->emplace_state<UniverseNumberParticleState>(count);
    } else if (state->size() != count) {
        state->data().resize(count);
    }

    if (auto* state = this->_universe->state<UniverseMaxRelativeSpeedState>();
        state == nullptr) {
        this->_universe->emplace_state<UniverseMaxRelativeSpeedState>(count);
    } else if (state->size() != count) {
        state->data().resize(count);
    }

    if (auto* state = this->_universe->state<UniverseMaxSigmaGState>();
        state == nullptr) {
        this->_universe->emplace_state<UniverseMaxSigmaGState>(count);
    } else if (state->size() != count) {
        state->data().resize(count);
    }

    if (auto* state = this->_universe->state<UniverseCollisionRemainderState>();
        state == nullptr) {
        this->_universe->emplace_state<UniverseCollisionRemainderState>(count);
    } else if (state->size() != count) {
        state->data().resize(count);
    }

    if (auto* state = this->_universe->state<UniverseCollisionCountState>();
        state == nullptr) {
        this->_universe->emplace_state<UniverseCollisionCountState>(count);
    } else if (state->size() != count) {
        state->data().resize(count);
    }
}

void
DsmcSolver::reset_states() {
    if (!this->_universe) {
        return;
    }

    ensure_states();
    this->_universe->state<UniverseNumberParticleState>()->reset();
    this->_universe->state<UniverseMaxRelativeSpeedState>()->reset();
    this->_universe->state<UniverseMaxSigmaGState>()->reset();
    this->_universe->state<UniverseCollisionRemainderState>()->reset();
    this->_universe->state<UniverseCollisionCountState>()->reset();
}

void
DsmcSolver::make_probe() noexcept {
    static_cast<void>(atlas::detail::DsmcProbeBuilder::make(
        _probe,
        this->_universe,
        this->_fluid,
        this->_searcher,
        _kernel,
        _collision_seed++));
}

bool
DsmcSolver::measure_collision_statistics(const DeviceBuffer<int>* allocated_solver,
                                         const int index,
                                         const float dt) {
    return _statistics.measure(_probe, allocated_solver, index, dt);
}

DsmcKernelType
DsmcSolver::kernel_type() const noexcept {

    return _kernel.type;
}

DsmcCollisionWorkloadType
DsmcSolver::workload_type() const noexcept {
    return _workload_type;
}

void
DsmcSolver::set_workload_type(const DsmcCollisionWorkloadType workload_type) noexcept {
    _workload_type = workload_type;
}

void
DsmcSolver::apply_flattened_collision(const DeviceBuffer<int>* allocated_solver,
                                      const int index) {
    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    if (!_flatten_workload.build(_probe.collision_count_ptr, _probe.cell_count, allocated_solver_ptr, index)) {
        return;
    }

    launch_flattened_collisions();
}

void
DsmcSolver::launch_flattened_collisions() {
    const auto probe                    = _probe;
    const int* collision_offsets_ptr    = atlas::raw_pointer_cast(_flatten_workload.collision_offsets.data());
    const int* collision_cells_ptr      = atlas::raw_pointer_cast(_flatten_workload.collision_cells.data());
    const int flattened_collision_count = _flatten_workload.flattened_collision_count;

    atlas::parallel_for<atlas::ExecutionPolicy::device>(
        0,
        flattened_collision_count,
        [=] ATLAS_ALL_DEVICE(const int work_index) {
            const int cell = collision_cells_ptr[work_index];
            // work_index is a global index into the flattened candidate
            // list; recover this candidate's position *within its own
            // cell* by subtracting that cell's starting offset, matching
            // the (cell, local_collision) stream key the per-cell path uses
            // so both workload types draw identical, reproducible samples.
            const int local_collision = work_index - collision_offsets_ptr[cell];
            const int count           = static_cast<int>(probe.number_particle_ptr[cell]);
            const float max_sigma_g   = probe.max_sigma_g_ptr[cell];
            const int begin           = probe.cell_start_ptr[cell];
            const int end             = probe.cell_end_ptr[cell];
            if (count < 2 || !(max_sigma_g > 0.0f) || begin < 0 || end <= begin) {
                return;
            }
            // Defensive: number_particle_ptr[cell] (from the last statistics
            // pass) could be stale relative to the searcher's current
            // cell_start/cell_end range if particles moved/were
            // added/removed between measure and apply; skip rather than
            // read past the cell's actual sorted-index range.
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

            DsmcSolver::collide_indexed_pair(
                probe,
                cell,
                stream,
                probe.indices_ptr[begin + lhs_local],
                probe.indices_ptr[begin + rhs_local],
                max_sigma_g);
        });
}

void
DsmcSolver::apply_collision(const DeviceBuffer<int>* allocated_solver,
                            const int index,
                            const float) {
    // dt is unused here: collision_count_ptr already bakes dt into its
    // expected-candidate-count computation (see DsmcStatistics::measure),
    // so applying candidates only needs the counts already measured.
    if (_workload_type == DsmcCollisionWorkloadType::flatten) {
        apply_flattened_collision(allocated_solver, index);
        return;
    }

    const int* allocated_solver_ptr = allocated_solver != nullptr
        ? atlas::raw_pointer_cast(allocated_solver->data())
        : nullptr;

    apply_cell_collisions(allocated_solver_ptr, index);
}

void
DsmcSolver::apply_cell_collisions(const int* allocated_solver_ptr, const int index) {
    const auto probe = _probe;

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
            // One deterministic hash stream per (cell, local_collision) pair
            // — see DsmcSolver's file-level docs for why this replaces a
            // per-thread RNG stream.
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

                DsmcSolver::collide_indexed_pair(
                    probe,
                    cell,
                    stream,
                    probe.indices_ptr[begin + lhs_local],
                    probe.indices_ptr[begin + rhs_local],
                    max_sigma_g);
            }
        });
}

}
