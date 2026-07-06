#include <atlas/orchestrator/orchestrator.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe_state.h>

#include <stdexcept>
#include <utility>

namespace atlas {

Orchestrator::Orchestrator(UniverseHostPtr universe,
                           FluidHostPtr fluid,
                           SearcherHostPtr searcher,
                           CodecHostPtr codec,
                           MeasurerHostPtr measurer,
                           HostBuffer<SolveHostPtr> solvers) noexcept
    : _universe(std::move(universe))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher))
    , _codec(std::move(codec))
    , _measurer(std::move(measurer))
    , _solvers(std::move(solvers)) {
}

Orchestrator::Builder
Orchestrator::builder() noexcept {
    return Builder {};
}

void
Orchestrator::search() {

    if (_searcher) {
        _searcher->build();
    }
}

void
Orchestrator::classify() {

    if (_codec) {
        _codec->update();
    }
}

void
Orchestrator::measure() {

    if (_measurer) {
        _measurer->measure();
    }
}

void
Orchestrator::measure(const float dt) {

    if (_measurer) {
        _measurer->measure(dt);
    }
}

void
Orchestrator::solve(const float dt) {

    if (_solvers.empty()) {
        return;
    }

    if (!_codec) {
        for (const auto& solver : _solvers) {
            if (!solver) continue;

            solver->solve(dt);
        }

        return;
    }

    const auto* allocated_solver = &_codec->allocated_solver();
    const int solver_count       = static_cast<int>(_solvers.size());

    for (int i = 0; i < solver_count; i++) {
        if (!_solvers[i]) continue;

        _solvers[i]->solve(allocated_solver, i, dt);
    }
}

void
Orchestrator::update(const float dt) {

    orchestrate(dt);
}

namespace {

bool
has_gravity(const OrchestratorProbe& probe) noexcept {
    return probe.gravity_ptr != nullptr && probe.gravity_cell_count > 0;
}

bool
has_field_force(const OrchestratorProbe& probe) noexcept {
    return probe.field_force_ptr != nullptr
        && probe.field_force_cell_count > 0
        && probe.species_ptr != nullptr
        && probe.properties_ptr != nullptr
        && probe.species_count > 0;
}

int
force_cell_count(const OrchestratorProbe& probe, const bool gravity, const bool field_force) noexcept {
    int cell_count = gravity ? probe.gravity_cell_count : 0;

    if (field_force && probe.field_force_cell_count > cell_count) {
        cell_count = probe.field_force_cell_count;
    }

    if (cell_count > probe.cell_count) {
        cell_count = probe.cell_count;
    }

    return cell_count;
}

}

void
Orchestrator::apply_forces(const OrchestratorProbe& probe, const float dt) {
    const bool gravity     = has_gravity(probe);
    const bool field_force = has_field_force(probe);

    if (!gravity && !field_force) {
        return;
    }

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        force_cell_count(probe, gravity, field_force),
        [=] ATLAS_ALL_DEVICE(const int cell) {
            const bool gravity_cell     = gravity && cell < probe.gravity_cell_count;
            const bool field_force_cell = field_force && cell < probe.field_force_cell_count;

            if (!gravity_cell && !field_force_cell) {
                return;
            }

            const int start = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (start < 0 || end <= start) {
                return;
            }

            const Float3 gravity_value = gravity_cell ? probe.gravity_ptr[cell] : Float3();
            const Float3 force_value   = field_force_cell ? probe.field_force_ptr[cell] : Float3();

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                Float3 delta_velocity;
                bool update_velocity = false;

                if (gravity_cell) {
                    delta_velocity += gravity_value * dt;
                    update_velocity = true;
                }

                if (field_force_cell) {
                    const std::size_t species_index = probe.species_ptr[particle_index];

                    if (species_index < static_cast<std::size_t>(probe.species_count)) {
                        const float mass = probe.properties_ptr[species_index].mass;

                        if (mass > 0.0f) {
                            delta_velocity += force_value * (dt / mass);
                            update_velocity = true;
                        }
                    }
                }

                if (update_velocity) {
                    probe.velocity_ptr[particle_index] += delta_velocity;
                }
            }
        });
}

void
Orchestrator::apply_field_force(const OrchestratorProbe& probe, const float dt) {
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.field_force_cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            const int start = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (start < 0 || end <= start) {
                return;
            }

            const Float3 force = probe.field_force_ptr[cell];

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                const std::size_t species_index = probe.species_ptr[particle_index];

                if (species_index >= static_cast<std::size_t>(probe.species_count)) {
                    continue;
                }

                const float mass = probe.properties_ptr[species_index].mass;

                if (!(mass > 0.0f)) {
                    continue;
                }

                probe.velocity_ptr[particle_index] += force * (dt / mass);
            }
        });
}

void
Orchestrator::apply_gravity(const OrchestratorProbe& probe, const float dt) {
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.gravity_cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            const int start = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            if (start < 0 || end <= start) {
                return;
            }

            const Float3 cell_gravity = probe.gravity_ptr[cell];

            for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                const int particle_index = probe.indices_ptr[sorted_index];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                probe.velocity_ptr[particle_index] += cell_gravity * dt;
            }
        });
}

void
Orchestrator::apply_probe_forces(const float dt) {
    const auto probe = _probe;
    apply_forces(probe, dt);
}

namespace {

bool
load_common_data(const UniverseHostPtr& universe,
                 const FluidHostPtr& fluid,
                 const SearcherHostPtr& searcher,
                 OrchestratorProbe& probe) noexcept {
    if (!universe || !fluid || !searcher) {
        return false;
    }

    auto* velocity_state = fluid->state<atlas::FluidVelocityState>();

    if (velocity_state == nullptr) {
        return false;
    }

    auto& velocity = velocity_state->data();

    probe.particle_count = static_cast<int>(fluid->particle_count());
    probe.cell_count     = universe->cell_count();
    probe.velocity_ptr   = atlas::raw_pointer_cast(velocity.data());

    probe.indices_ptr    = searcher->indices();
    probe.cell_start_ptr = searcher->cell_start();
    probe.cell_end_ptr   = searcher->cell_end();

    return !velocity.empty()
        && probe.particle_count > 0
        && probe.cell_count > 0
        && probe.indices_ptr != nullptr
        && probe.cell_start_ptr != nullptr
        && probe.cell_end_ptr != nullptr;
}

void
load_species_data(const FluidHostPtr& fluid, OrchestratorProbe& probe) noexcept {
    auto* species_state = fluid->state<atlas::FluidSpeciesState>();

    if (species_state == nullptr) {
        return;
    }

    auto& species             = species_state->data();
    auto& particle_properties = fluid->particle_properties();

    if (species.empty() || particle_properties.empty()) {
        return;
    }

    probe.species_ptr    = atlas::raw_pointer_cast(species.data());
    probe.properties_ptr = atlas::raw_pointer_cast(particle_properties.data());
    probe.species_count  = static_cast<int>(particle_properties.size());
}

void
load_field_force_data(const UniverseHostPtr& universe, OrchestratorProbe& probe) noexcept {
    auto* field_force_state = universe->state<atlas::UniverseFieldForceState>();

    if (field_force_state == nullptr) {
        return;
    }

    auto& field_force = field_force_state->data();

    if (field_force.empty()) {
        return;
    }

    probe.field_force_ptr        = atlas::raw_pointer_cast(field_force.data());
    probe.field_force_cell_count = static_cast<int>(field_force.size());
}

void
load_gravity_data(const UniverseHostPtr& universe, OrchestratorProbe& probe) noexcept {
    auto* gravity_state = universe->state<atlas::UniverseGravityState>();

    if (gravity_state == nullptr) {
        return;
    }

    auto& gravity = gravity_state->data();

    if (gravity.empty()) {
        return;
    }

    probe.gravity_ptr        = atlas::raw_pointer_cast(gravity.data());
    probe.gravity_cell_count = static_cast<int>(gravity.size());
}

}

bool
Orchestrator::make_probe() noexcept {
    _probe = {};

    if (!load_common_data(_universe, _fluid, _searcher, _probe)) {
        return false;
    }

    load_species_data(_fluid, _probe);
    load_field_force_data(_universe, _probe);
    load_gravity_data(_universe, _probe);

    return true;
}

void
Orchestrator::orchestrate(const float dt) {
    if (searcher()) {
        searcher()->invalidate();
    }
    search();
    classify();
    measure(dt);

    if (dt != 0.0f && make_probe()) {
        apply_probe_forces(dt);
    }

    solve(dt);
}

void
Orchestrator::set_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
}

void
Orchestrator::set_fluid(FluidHostPtr fluid) noexcept {
    _fluid = std::move(fluid);
}

void
Orchestrator::set_searcher(SearcherHostPtr searcher) noexcept {
    _searcher = std::move(searcher);
}

void
Orchestrator::set_codec(CodecHostPtr codec) noexcept {
    _codec = std::move(codec);
}

void
Orchestrator::set_measurer(MeasurerHostPtr measurer) noexcept {
    _measurer = std::move(measurer);
}

void
Orchestrator::add_solver(SolveHostPtr solver) noexcept {
    _solvers.push_back(std::move(solver));
}

const SearcherHostPtr&
Orchestrator::searcher() const noexcept {
    return _searcher;
}

const UniverseHostPtr&
Orchestrator::universe() const noexcept {
    return _universe;
}

const FluidHostPtr&
Orchestrator::fluid() const noexcept {
    return _fluid;
}

const CodecHostPtr&
Orchestrator::codec() const noexcept {
    return _codec;
}

const MeasurerHostPtr&
Orchestrator::measurer() const noexcept {
    return _measurer;
}

const HostBuffer<SolveHostPtr>&
Orchestrator::solvers() const noexcept {
    return _solvers;
}

Orchestrator::Builder&
Orchestrator::Builder::with_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

Orchestrator::Builder&
Orchestrator::Builder::with_fluid(FluidHostPtr fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

Orchestrator::Builder&
Orchestrator::Builder::with_searcher(SearcherHostPtr searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

Orchestrator::Builder&
Orchestrator::Builder::with_codec(CodecHostPtr codec) noexcept {
    _codec = std::move(codec);
    return *this;
}

Orchestrator::Builder&
Orchestrator::Builder::with_measurer(MeasurerHostPtr measurer) noexcept {
    _measurer = std::move(measurer);
    return *this;
}

Orchestrator::Builder&
Orchestrator::Builder::with_gravity(const Float3& gravity) noexcept {
    _gravity = gravity;
    return *this;
}

Orchestrator::Builder&
Orchestrator::Builder::with_solver(SolveHostPtr solver) noexcept {
    _solvers.push_back(std::move(solver));
    return *this;
}

void
Orchestrator::Builder::validate() const {

    for (const auto& solver : _solvers) {
        if (!solver) {
            throw std::runtime_error("Orchestrator::Builder: solver must not be null.");
        }
    }
}

void
Orchestrator::Builder::ensure_gravity_state() const {

    if (!_universe || !_gravity.has_value()) {
        return;
    }

    const std::size_t cell_count = static_cast<std::size_t>(_universe->cell_count());

    if (cell_count == 0) {
        return;
    }

    if (_universe->has_state<atlas::UniverseGravityState>()) {
        auto* gravity_state = _universe->state<atlas::UniverseGravityState>();

        if (gravity_state != nullptr) {
            if (gravity_state->size() != cell_count) {
                gravity_state->data().resize(cell_count);
            }

            atlas::parallel_fill<ExecutionPolicy::device>(
                gravity_state->data().begin(),
                gravity_state->data().end(),
                *_gravity);
        }

        return;
    }

    _universe->emplace_state<atlas::UniverseGravityState>(cell_count);

    auto* gravity_state = _universe->state<atlas::UniverseGravityState>();

    if (gravity_state == nullptr) {
        return;
    }

    atlas::parallel_fill<ExecutionPolicy::device>(
        gravity_state->data().begin(),
        gravity_state->data().end(),
        *_gravity);
}

Orchestrator
Orchestrator::Builder::build() const {
    validate();
    ensure_gravity_state();

    return Orchestrator(_universe, _fluid, _searcher, _codec, _measurer, _solvers);
}

atlas::host_shared_ptr<Orchestrator>
Orchestrator::Builder::make_host_shared() const {
    validate();
    ensure_gravity_state();

    return atlas::make_host_shared<Orchestrator>(
        _universe,
        _fluid,
        _searcher,
        _codec,
        _measurer,
        _solvers);
}

}
