#include <atlas/system/system.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/atomic.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/serialization/protobuf_snapshot.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>
#include <atlas/universe/universe_state.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace atlas {

namespace {

    /**
     * @brief Guarantees a universe holds a state column of type @p StateT sized to the grid.
     *
     * Emplaces the column when absent, otherwise resizes it in place if its length no longer
     * matches the cell count. Lets @c System provision exactly the per-cell columns its
     * configured solvers require, idempotently.
     *
     * @tparam StateT   The universe state column type to ensure.
     * @param universe   The universe whose state store is amended.
     * @param cell_count Number of cells the column must cover.
     */
    template <typename StateT>
    void
    ensure_universe_state(Universe& universe, const std::size_t cell_count) {
        auto* state = universe.state<StateT>();

        if (state == nullptr) {
            universe.emplace_state<StateT>(cell_count);
            return;
        }

        if (state->size() != cell_count) {
            state->data().resize(cell_count);
        }
    }

}

System::System(FluidHostPtr fluid,
               UniverseHostPtr universe,
               HostBuffer<SolverHostPtr> solvers,
               HostBuffer<SourceHostPtr> sources,
               HostBuffer<GeneratorHostPtr> generators,
               const HostBuffer<Collider>& colliders,
               const HostBuffer<Sink>& sinks,
               CodecHostPtr codec,
               const float dt)
    : _dt(dt)
    , _fluid(std::move(fluid))
    , _universe(std::move(universe))
    , _solvers(std::move(solvers))
    , _codec(std::move(codec))
    , _sources(std::move(sources))
    , _generators(std::move(generators))
    , _source_spawned_last_step(_sources.size(), 0)
    , _colliders(colliders.begin(), colliders.end())
    , _sinks(sinks.begin(), sinks.end())
    , _sink_removed_last_step(_sinks.size(), 0) {

    // The searcher indexes the universe grid, so it can only exist once a universe does.
    if (_universe) {
        _searcher = SpatialHashingSearcher::builder()
                        .with_universe(*_universe)
                        .make_host_shared();
    }

    initialize_states();
}

void
System::initialize_states() {
    if (!_universe) {
        return;
    }

    const auto cell_count = static_cast<std::size_t>(_universe->cell_count());

    // Every configuration needs the per-cell solver-selection column, regardless of solver.
    ensure_universe_state<UniverseAllocatedSolverState>(*_universe, cell_count);

    // Each solver kind pulls in the extra per-cell columns it reads or writes.
    for (const auto& solver : _solvers) {
        switch (solver->type()) {
        case SolverType::dsmc:
            initialize_dsmc_states();
            break;
        }
    }
}

void
System::initialize_dsmc_states() {
    const auto cell_count = static_cast<std::size_t>(_universe->cell_count());

    ensure_universe_state<UniverseNumberParticleState>(*_universe, cell_count);
    ensure_universe_state<UniverseMaxRelativeSpeedState>(*_universe, cell_count);
    ensure_universe_state<UniverseMaxSigmaGState>(*_universe, cell_count);
    ensure_universe_state<UniverseCollisionCountState>(*_universe, cell_count);
}

System::Builder
System::builder() noexcept {
    return Builder {};
}

std::string
System::snapshot_directory_name(const std::size_t step) {
    return "time_step_" + std::to_string(step);
}

void
System::save(const std::filesystem::path& directory) const {
    const std::filesystem::path step_directory = directory / snapshot_directory_name(_step);

    std::filesystem::create_directories(step_directory);

    atlas::save_fluid_binary(*_fluid, (step_directory / "fluid.bin").string());
    atlas::save_universe_binary(*_universe, (step_directory / "universe.bin").string());
}

void
System::update() {
    emit();
    search();
    allocate();
    solve();
    advect();
    remove();

    ++_step;
}

void
System::emit() {
    std::fill(_source_spawned_last_step.begin(), _source_spawned_last_step.end(), 0);

    auto* positions = _fluid->state<FluidPositionState>();

    if (positions == nullptr) {
        return;
    }

    auto* velocities = _fluid->state<FluidVelocityState>();
    auto* species    = _fluid->state<FluidSpeciesState>();

    const std::size_t buffer_size = _fluid->buffer_size();

    // New particles are appended at the current live count and grow it in place.
    std::size_t count = _fluid->particle_count();

    for (std::size_t i = 0; i < _sources.size(); ++i) {
        // Stop once the fixed-capacity fluid buffer is full; excess spawns are dropped.
        if (count >= buffer_size) {
            break;
        }

        const int spawned = _sources[i]->spawn(positions, count);
        _source_spawned_last_step[i] = spawned > 0 ? spawned : 0;

        if (spawned <= 0) {
            continue;
        }

        // The paired generator fills velocity and species for exactly the spawned slots;
        // its returned count is unused because `spawned` already fixes the slot range.
        static_cast<void>(_generators[i]->generate(
            velocities,
            species,
            count,
            static_cast<std::size_t>(spawned)));

        count += static_cast<std::size_t>(spawned);
    }

    _fluid->set_particle_count(count);

    // Advance the emitter boundaries after emission so the next step sees moved sources.
    for (const auto& source : _sources) {
        source->advance(_dt);
    }
}

void
System::search() {
    if (!_searcher) {
        return;
    }

    _searcher->classify(
        _fluid->state<FluidPositionState>(),
        _universe->state<UniverseNumberParticleState>(),
        static_cast<int>(_fluid->particle_count()));
}

void
System::allocate() {
    if (!_codec) {
        return;
    }

    _codec->allocate(
        _universe->state<UniverseTemperatureState>(),
        _universe->state<UniverseNumberParticleState>(),
        _universe->state<UniverseAllocatedSolverState>());
}

void
System::solve() {
    if (!_fluid || !_universe || !_searcher || _solvers.empty()) {
        return;
    }

    // Gather the trivially-copyable searcher view once so device kernels can capture it.
    const SpatialHashingSearcherView searcher_view = _searcher->view();

    // The loop index doubles as the solver's id, which cells compare against their
    // allocated-solver selection to decide whose work applies to them.
    for (std::size_t i = 0; i < _solvers.size(); ++i) {
        _solvers[i]->solve(*_fluid, *_universe, searcher_view, static_cast<int>(i), _dt);
    }
}

void
System::advect() {
    auto* position_state = _fluid->state<FluidPositionState>();
    auto* velocity_state = _fluid->state<FluidVelocityState>();

    if (position_state == nullptr || velocity_state == nullptr) {
        return;
    }

    const int particle_count = static_cast<int>(_fluid->particle_count());
    const int collider_count = static_cast<int>(_colliders.size());
    const float dt           = _dt;

    if (particle_count > 0) {
        auto* positions       = atlas::raw_pointer_cast(position_state->data().data());
        auto* velocities      = atlas::raw_pointer_cast(velocity_state->data().data());
        const auto* colliders = atlas::raw_pointer_cast(_colliders.data());

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            particle_count,
            [=] ATLAS_ALL_DEVICE(const int i) {
                Float3 position = positions[i];
                Float3 velocity = velocities[i];

                const float sweep_length = velocity.length() * dt;

                // A particle that barely moves this step cannot cross a boundary; skip the
                // collision search entirely and leave its position untouched.
                if (sweep_length <= atlas::eps) {
                    return;
                }

                // Broad phase: the swept segment against each collider's world
                // AABB. Narrow phase: trace only the survivors, keep the
                // nearest hit.
                const atlas::Ray sweep(position, velocity);

                HitSurface nearest {};
                int nearest_collider = -1;

                for (int c = 0; c < collider_count; ++c) {
                    // An unbounded geometry (a plane) has no valid world AABB,
                    // so it skips the broad phase rather than reject everything.
                    const AABB& bound = colliders[c].bound();

                    if (bound.is_valid()) {
                        const HitAABB coarse = bound.trace(sweep);

                        if (!coarse.is_intersecting || coarse.enter > sweep_length) {
                            continue;
                        }
                    }

                    const HitSurface hit = colliders[c].trace(position, velocity, dt);

                    if (hit.is_intersecting && hit.distance < nearest.distance) {
                        nearest          = hit;
                        nearest_collider = c;
                    }
                }

                // Resolve against the nearest hit (which updates position and velocity), or
                // integrate straight ahead when the swept segment hit nothing.
                if (nearest_collider >= 0) {
                    colliders[nearest_collider].collide(nearest, position, velocity, dt);
                } else {
                    position += velocity * dt;
                }

                positions[i]  = position;
                velocities[i] = velocity;
            });
    }

    // Advance the collider boundaries in a separate pass, after particles have been traced
    // against their positions for this step.
    if (collider_count > 0) {
        auto* colliders = atlas::raw_pointer_cast(_colliders.data());

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            collider_count,
            [=] ATLAS_ALL_DEVICE(const int c) {
                colliders[c].advance(dt);
            });
    }
}

void
System::mark_survivors(const int particle_count) {
    auto* position_state = _fluid->state<FluidPositionState>();
    auto* velocity_state = _fluid->state<FluidVelocityState>();

    const auto* positions  = atlas::raw_pointer_cast(position_state->data().data());
    const auto* velocities = atlas::raw_pointer_cast(velocity_state->data().data());
    auto* active           = atlas::raw_pointer_cast(_fluid->active().data());

    const auto* sinks    = atlas::raw_pointer_cast(_sinks.data());
    auto* removed        = atlas::raw_pointer_cast(_sink_removed_last_step.data());
    const int sink_count = static_cast<int>(_sinks.size());
    const float dt       = _dt;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_count,
        [=] ATLAS_ALL_DEVICE(const int i) {
            const Float3 position = positions[i];
            const Float3 velocity = velocities[i];

            // First sink to claim the particle wins: flag it dead and stop.
            for (int s = 0; s < sink_count; ++s) {
                if (sinks[s].despawn(position, velocity, dt)) {
                    atlas::atomic_add(removed + s, 1);
                    active[i] = 0;
                    return;
                }
            }

            // Survived every sink: mark alive so compact() keeps it.
            active[i] = 1;
        });
}

void
System::remove() {
    const int particle_count = static_cast<int>(_fluid->particle_count());
    const int sink_count     = static_cast<int>(_sinks.size());

    atlas::parallel_fill<ExecutionPolicy::device>(
        _sink_removed_last_step.begin(),
        _sink_removed_last_step.end(),
        0);

    if (sink_count > 0 && particle_count > 0
        && _fluid->state<FluidPositionState>() != nullptr
        && _fluid->state<FluidVelocityState>() != nullptr) {

        mark_survivors(particle_count);

        // The fluid owns the scan, the gather indices, and every state it has
        // to gather.
        static_cast<void>(_fluid->compact());
    }

    // Advance the sink boundaries even when nothing was removed this step (they may still be
    // moving), so long as any sinks exist.
    if (sink_count > 0) {
        auto* sinks    = atlas::raw_pointer_cast(_sinks.data());
        const float dt = _dt;

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            sink_count,
            [=] ATLAS_ALL_DEVICE(const int s) {
                sinks[s].advance(dt);
            });
    }
}

std::vector<int>
System::sink_removed_last_step() const {
    std::vector<int> counts(_sink_removed_last_step.size());
    atlas::copy_device_to_host(_sink_removed_last_step, counts.data(), counts.size());
    return counts;
}

System::Builder&
System::Builder::with_fluid(FluidHostPtr fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

System::Builder&
System::Builder::with_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

System::Builder&
System::Builder::with_solver(SolverHostPtr solver) {
    _solvers.push_back(std::move(solver));
    return *this;
}

System::Builder&
System::Builder::with_emitter(SourceHostPtr source, GeneratorHostPtr generator) {
    _sources.push_back(std::move(source));
    _generators.push_back(std::move(generator));
    return *this;
}

System::Builder&
System::Builder::with_collider(const Collider& collider) {
    _colliders.push_back(collider);
    return *this;
}

System::Builder&
System::Builder::with_sink(const Sink& sink) {
    _sinks.push_back(sink);
    return *this;
}

System::Builder&
System::Builder::with_codec(CodecHostPtr codec) noexcept {
    _codec = std::move(codec);
    return *this;
}

System::Builder&
System::Builder::with_dt(const float dt) noexcept {
    _dt = dt;
    return *this;
}

void
System::Builder::validate() const {
    if (!_fluid) {
        throw std::runtime_error("System::Builder: fluid must not be null.");
    }

    if (!_universe) {
        throw std::runtime_error("System::Builder: universe must not be null.");
    }

    if (!(_dt > 0.0f)) {
        throw std::runtime_error("System::Builder: dt must be positive.");
    }

    if (_sources.size() != _generators.size()) {
        throw std::runtime_error("System::Builder: every source needs a paired generator.");
    }

    for (std::size_t i = 0; i < _sources.size(); ++i) {
        if (!_sources[i] || !_generators[i]) {
            throw std::runtime_error("System::Builder: source and generator must not be null.");
        }
    }

    for (const auto& solver : _solvers) {
        if (!solver) {
            throw std::runtime_error("System::Builder: a solver must not be null.");
        }
    }
}

System
System::Builder::build() {
    validate();

    return System(std::move(_fluid),
                  std::move(_universe),
                  _solvers,
                  std::move(_sources),
                  std::move(_generators),
                  _colliders,
                  _sinks,
                  _codec,
                  _dt);
}

atlas::host_unique_ptr<System>
System::Builder::make_host_unique() {
    return atlas::make_host_unique<System>(build());
}

}
