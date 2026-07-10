#include <atlas/system/system.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/atomic.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/serialization/protobuf_snapshot.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>
#include <atlas/universe/universe_state.h>

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
               ObserverHostPtr observer,
               const float dt)
    : _dt(dt)
    , _fluid(std::move(fluid))
    , _universe(std::move(universe))
    , _solvers(std::move(solvers))
    , _codec(std::move(codec))
    , _observer(std::move(observer))
    , _sources(std::move(sources))
    , _generators(std::move(generators))
    , _colliders(colliders.begin(), colliders.end())
    , _sinks(sinks.begin(), sinks.end()) {

    // The searcher indexes the universe grid, so it can only exist once a universe does.
    if (_universe) {
        _searcher = SpatialHashingSearcher::builder()
                        .with_universe(*_universe)
                        .make_host_shared();
    }

    if (_observer) {
        // Without a material dictionary there is a single implicit species; the counter
        // layout must reserve at least one column so observer bookkeeping stays valid.
        const std::size_t species_count
            = (_fluid && _fluid->materials()) ? _fluid->materials()->size() : std::size_t { 1 };

        _observer->resize_counters(_sources.size(), _sinks.size(), species_count);
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

    if (_observer) {
        _observer->observe(*_fluid, *_universe, _step);
    }
}

void
System::emit() {
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

        // Record the spawn against the range starting at the pre-increment offset.
        record_spawned(i, count, static_cast<std::size_t>(spawned));

        count += static_cast<std::size_t>(spawned);
    }

    _fluid->set_particle_count(count);

    // Advance the emitter boundaries after emission so the next step sees moved sources.
    for (const auto& source : _sources) {
        source->advance(_dt);
    }
}

void
System::record_spawned(const std::size_t source_index, const std::size_t offset, const std::size_t count) {
    if (!_observer || count == 0) {
        return;
    }

    const auto* species_state       = _fluid->state<FluidSpeciesState>();
    const std::size_t species_count = _observer->species_count();

    // Bail unless the species column exists and the counter matrix has its expected
    // (source_count x species_count) shape; a mismatch means the layout is stale.
    if (species_state == nullptr || species_count == 0
        || _observer->spawned().size() != _sources.size() * species_count) {
        return;
    }

    const auto* species = atlas::raw_pointer_cast(species_state->data().data());
    auto* spawned       = atlas::raw_pointer_cast(_observer->spawned().data());

    // Row offset of this source's counters within the flattened matrix.
    const auto base          = static_cast<int>(source_index * species_count);
    const auto species_limit = static_cast<int>(species_count);
    const auto first         = static_cast<int>(offset);

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(count),
        [=] ATLAS_ALL_DEVICE(const int k) {
            const auto id = static_cast<int>(species[first + k]);

            // Guard the species id before indexing; unknown ids are simply not counted.
            if (id >= 0 && id < species_limit) {
                atlas::atomic_add(spawned + base + id, 1);
            }
        });
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
    const int sink_count = static_cast<int>(_sinks.size());
    const float dt       = _dt;

    const auto* species_state = _fluid->state<FluidSpeciesState>();

    // Despawn tallying is optional: these stay null unless an observer, a species column,
    // and a correctly shaped (sink_count x species_count) counter matrix are all present,
    // in which case the kernel below counts removals per sink and species.
    int* despawned             = nullptr;
    const std::size_t* species = nullptr;
    int species_count          = 0;

    if (_observer && species_state != nullptr
        && _observer->despawned().size() == _sinks.size() * _observer->species_count()) {

        despawned     = atlas::raw_pointer_cast(_observer->despawned().data());
        species       = atlas::raw_pointer_cast(species_state->data().data());
        species_count = static_cast<int>(_observer->species_count());
    }

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_count,
        [=] ATLAS_ALL_DEVICE(const int i) {
            const Float3 position = positions[i];
            const Float3 velocity = velocities[i];

            // First sink to claim the particle wins: flag it dead, tally it, and stop.
            for (int s = 0; s < sink_count; ++s) {
                if (sinks[s].despawn(position, velocity, dt)) {
                    active[i] = 0;

                    if (despawned != nullptr) {
                        const auto id = static_cast<int>(species[i]);

                        // Guard the species id before indexing the counter row for sink s.
                        if (id >= 0 && id < species_count) {
                            atlas::atomic_add(despawned + s * species_count + id, 1);
                        }
                    }

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
System::Builder::with_observer(ObserverHostPtr observer) noexcept {
    _observer = std::move(observer);
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
                  _observer,
                  _dt);
}

atlas::host_unique_ptr<System>
System::Builder::make_host_unique() {
    return atlas::make_host_unique<System>(build());
}

}
