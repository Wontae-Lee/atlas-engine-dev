#include <atlas/system/system.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <stdexcept>
#include <utility>

namespace atlas {

System::System(FluidHostPtr fluid,
               UniverseHostPtr universe,
               HostBuffer<SourceHostPtr> sources,
               HostBuffer<GeneratorHostPtr> generators,
               const HostBuffer<Collider>& colliders,
               const float dt)
    : _dt(dt)
    , _fluid(std::move(fluid))
    , _universe(std::move(universe))
    , _sources(std::move(sources))
    , _generators(std::move(generators))
    , _colliders(colliders.begin(), colliders.end()) {

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

    return System(_fluid, _universe, _sources, _generators, _colliders, _dt);
}

atlas::host_shared_ptr<System>
System::Builder::make_host_shared() const {
    return atlas::make_host_shared<System>(build());
}

}
