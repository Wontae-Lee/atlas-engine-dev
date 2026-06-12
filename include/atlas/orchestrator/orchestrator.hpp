#pragma once

#include <atlas/parallel/parallel_fill.h>
#include <atlas/universe/universe_state.h>

#include <stdexcept>
#include <utility>

namespace atlas {

template <typename T>
Orchestrator<T>::Orchestrator(UniverseHostPtr<T> universe,
                              FluidHostPtr<T> fluid,
                              SpatialHashingSearcherHostPtr<T> searcher,
                              CodecHostPtr<T> codec,
                              MeasurerHostPtr<T> measurer,
                              HostBuffer<SolveHostPtr<T>> solvers) noexcept
    : _universe(std::move(universe))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher))
    , _codec(std::move(codec))
    , _measurer(std::move(measurer))
    , _solvers(std::move(solvers)) {
}

template <typename T>
typename Orchestrator<T>::Builder
Orchestrator<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Orchestrator<T>::search() {

    if (_searcher) {
        _searcher->build();
    }
}

template <typename T>
void
Orchestrator<T>::classify() {

    if (_codec) {
        _codec->update();
    }
}

template <typename T>
void
Orchestrator<T>::measure() {

    if (_measurer) {
        _measurer->measure();
    }
}

template <typename T>
void
Orchestrator<T>::measure(const T dt) {

    if (_measurer) {
        _measurer->measure(dt);
    }
}

template <typename T>
void
Orchestrator<T>::solve(const T dt) {

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

template <typename T>
void
Orchestrator<T>::update(const T dt) {

    orchestrate(dt);
}

template <typename T>
void
Orchestrator<T>::apply_forces(const OrchestratorProbe<T>& probe, const T dt) {
    _force_applier.apply_all(probe, dt);
}

template <typename T>
void
Orchestrator<T>::apply_field_force(const OrchestratorProbe<T>& probe, const T dt) {
    _force_applier.apply_field_force(probe, dt);
}

template <typename T>
void
Orchestrator<T>::apply_gravity(const OrchestratorProbe<T>& probe, const T dt) {
    _force_applier.apply_gravity(probe, dt);
}

template <typename T>
void
Orchestrator<T>::apply_probe_forces(const T dt) {
    const auto probe = _probe;
    apply_forces(probe, dt);
}

template <typename T>
bool
Orchestrator<T>::make_probe() noexcept {
    return _probe_builder.build(_universe, _fluid, _searcher, _probe);
}

template <typename T>
void
Orchestrator<T>::orchestrate(const T dt) {

    _pipeline.run(*this, dt);
}

template <typename T>
void
Orchestrator<T>::set_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
}

template <typename T>
void
Orchestrator<T>::set_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
}

template <typename T>
void
Orchestrator<T>::set_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
}

template <typename T>
void
Orchestrator<T>::set_codec(CodecHostPtr<T> codec) noexcept {
    _codec = std::move(codec);
}

template <typename T>
void
Orchestrator<T>::set_measurer(MeasurerHostPtr<T> measurer) noexcept {
    _measurer = std::move(measurer);
}

template <typename T>
void
Orchestrator<T>::add_solver(SolveHostPtr<T> solver) noexcept {
    _solvers.push_back(std::move(solver));
}

template <typename T>
const SpatialHashingSearcherHostPtr<T>&
Orchestrator<T>::searcher() const noexcept {
    return _searcher;
}

template <typename T>
const UniverseHostPtr<T>&
Orchestrator<T>::universe() const noexcept {
    return _universe;
}

template <typename T>
const FluidHostPtr<T>&
Orchestrator<T>::fluid() const noexcept {
    return _fluid;
}

template <typename T>
const CodecHostPtr<T>&
Orchestrator<T>::codec() const noexcept {
    return _codec;
}

template <typename T>
const MeasurerHostPtr<T>&
Orchestrator<T>::measurer() const noexcept {
    return _measurer;
}

template <typename T>
const HostBuffer<SolveHostPtr<T>>&
Orchestrator<T>::solvers() const noexcept {
    return _solvers;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_codec(CodecHostPtr<T> codec) noexcept {
    _codec = std::move(codec);
    return *this;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_measurer(MeasurerHostPtr<T> measurer) noexcept {
    _measurer = std::move(measurer);
    return *this;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_gravity(const Vector3<T>& gravity) noexcept {
    _gravity = gravity;
    return *this;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_solver(SolveHostPtr<T> solver) noexcept {
    _solvers.push_back(std::move(solver));
    return *this;
}

template <typename T>
void
Orchestrator<T>::Builder::validate() const {

    for (const auto& solver : _solvers) {
        if (!solver) {
            throw std::runtime_error("Orchestrator::Builder: solver must not be null.");
        }
    }
}

template <typename T>
void
Orchestrator<T>::Builder::ensure_gravity_state() const {

    if (!_universe || !_gravity.has_value()) {
        return;
    }

    const std::size_t number_of_cells = _universe->number_of_cells();

    if (number_of_cells == 0) {
        return;
    }

    if (_universe->template has_state<atlas::UniverseGravityState<T>>()) {
        auto* gravity_state = _universe->template state<atlas::UniverseGravityState<T>>();

        if (gravity_state != nullptr) {
            if (gravity_state->size() != number_of_cells) {
                gravity_state->data().resize(number_of_cells);
            }

            atlas::parallel_fill<ExecutionPolicy::device>(
                gravity_state->data().begin(),
                gravity_state->data().end(),
                *_gravity);
        }

        return;
    }

    _universe->template emplace_state<atlas::UniverseGravityState<T>>(number_of_cells);

    auto* gravity_state = _universe->template state<atlas::UniverseGravityState<T>>();

    if (gravity_state == nullptr) {
        return;
    }

    atlas::parallel_fill<ExecutionPolicy::device>(
        gravity_state->data().begin(),
        gravity_state->data().end(),
        *_gravity);
}

template <typename T>
Orchestrator<T>
Orchestrator<T>::Builder::build() const {
    validate();
    ensure_gravity_state();

    return Orchestrator<T>(_universe, _fluid, _searcher, _codec, _measurer, _solvers);
}

template <typename T>
atlas::host_shared_ptr<Orchestrator<T>>
Orchestrator<T>::Builder::make_host_shared() const {
    validate();
    ensure_gravity_state();

    return atlas::make_host_shared<Orchestrator<T>>(
        _universe,
        _fluid,
        _searcher,
        _codec,
        _measurer,
        _solvers);
}

}