#include <atlas/system/system.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/serialization/protobuf_snapshot.h>
#include <atlas/universe/universe_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace atlas {

System::System(FluidHostPtr fluid,
               UniverseHostPtr universe,
               OrchestratorHostPtr orchestrator,
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
    , _orchestrator(std::move(orchestrator))
    , _codec(std::move(codec))
    , _observer(std::move(observer))
    , _sources(std::move(sources))
    , _generators(std::move(generators))
    , _colliders(colliders.begin(), colliders.end())
    , _sinks(sinks.begin(), sinks.end()) {

    if (_universe) {
        _searcher = SpatialHashingSearcher::builder()
                        .with_universe(*_universe)
                        .make_host_shared();
    }
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
    orchestrate();
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

    std::size_t count = _fluid->particle_count();

    for (std::size_t i = 0; i < _sources.size(); ++i) {
        if (count >= buffer_size) {
            break;
        }

        const int spawned = _sources[i]->spawn(positions, count);

        if (spawned <= 0) {
            continue;
        }

        static_cast<void>(_generators[i]->generate(
            velocities,
            species,
            count,
            static_cast<std::size_t>(spawned)));

        count += static_cast<std::size_t>(spawned);
    }

    _fluid->set_particle_count(count);

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
System::orchestrate() {
    if (!_orchestrator || !_searcher) {
        return;
    }

    _orchestrator->orchestrate(_searcher->view(), _dt);
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

                if (nearest_collider >= 0) {
                    colliders[nearest_collider].collide(nearest, position, velocity, dt);
                } else {
                    position += velocity * dt;
                }

                positions[i]  = position;
                velocities[i] = velocity;
            });
    }

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

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        particle_count,
        [=] ATLAS_ALL_DEVICE(const int i) {
            const Float3 position = positions[i];
            const Float3 velocity = velocities[i];

            for (int s = 0; s < sink_count; ++s) {
                if (sinks[s].despawn(position, velocity, dt)) {
                    active[i] = 0;
                    return;
                }
            }

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
System::Builder::with_orchestrator(OrchestratorHostPtr orchestrator) noexcept {
    _orchestrator = std::move(orchestrator);
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

}

System
System::Builder::build() const {
    validate();

    return System(_fluid, _universe, _orchestrator, _sources, _generators, _colliders, _sinks, _codec, _observer, _dt);
}

atlas::host_shared_ptr<System>
System::Builder::make_host_shared() const {
    return atlas::make_host_shared<System>(build());
}

}
