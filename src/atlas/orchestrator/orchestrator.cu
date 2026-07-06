#include <atlas/orchestrator/orchestrator.h>

#include <atlas/parallel/parallel_fill.h>
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

void
Orchestrator::apply_forces(const OrchestratorProbe& probe, const float dt) {
    _force_applier.apply_all(probe, dt);
}

void
Orchestrator::apply_field_force(const OrchestratorProbe& probe, const float dt) {
    _force_applier.apply_field_force(probe, dt);
}

void
Orchestrator::apply_gravity(const OrchestratorProbe& probe, const float dt) {
    _force_applier.apply_gravity(probe, dt);
}

void
Orchestrator::apply_probe_forces(const float dt) {
    const auto probe = _probe;
    apply_forces(probe, dt);
}

bool
Orchestrator::make_probe() noexcept {
    return detail::OrchestratorProbeBuilder::make(_universe, _fluid, _searcher, _probe);
}

void
Orchestrator::orchestrate(const float dt) {

    _pipeline.run(*this, dt);
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
Orchestrator::Builder::with_gravity(const Vector3& gravity) noexcept {
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
