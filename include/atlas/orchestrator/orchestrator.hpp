#pragma once

#include <stdexcept>

namespace atlas::system {

template <typename T>
Orchestrator<T>::Orchestrator(SpatialHashingSearcherHostPtr<T> searcher,
                              CodecHostPtr<T> codec,
                              MeasurerHostPtr<T> measurer,
                              HostBuffer<SolveHostPtr<T>> solvers) noexcept
    : _searcher(std::move(searcher))
    , _codec(std::move(codec))
    , _measurer(std::move(measurer))
    , _solvers(std::move(solvers)) {
    // Store the optional pipeline dependencies and the ordered solver list.
}

template <typename T>
typename Orchestrator<T>::Builder
Orchestrator<T>::builder() noexcept {

    // Return a default-initialized builder for fluent Orchestrator construction.
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
Orchestrator<T>::solve(const T dt) {

    // If there are no solvers, there is nothing to execute.
    if (_solvers.empty()) {
        return;
    }

    // When no codec is configured, execute each non-null solver through the
    // plain solve(dt) entry point with no codec-side orchestration context.
    if (!_codec) {

        for (const auto& solver : _solvers) {

            if (!solver) continue;

            solver->solve(dt);
        }

        return;
    }

    // When a codec is configured, retrieve the solver-allocation object owned
    // by the codec and pass it to every solver together with the solver index.
    //
    // This allows each solver to access codec-managed orchestration resources
    // while still knowing its position in the ordered solver sequence.
    const auto* allocated_solver = &_codec->allocated_solver();
    const int solver_count       = static_cast<int>(_solvers.size());
    for (int i = 0; i < solver_count; i++) {
        if (!_solvers[i]) continue;

        // Execute the solver in codec-aware mode.
        //
        // Parameters:
        // - allocated_solver : codec-owned solver allocation/context
        // - i                : current solver index in execution order
        _solvers[i]->solve(allocated_solver, i, dt);
    }
}

template <typename T>
void
Orchestrator<T>::orchestrate(const T dt) {

    search();
    classify();
    measure();
    solve(dt);
}

template <typename T>
void
Orchestrator<T>::set_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {

    _searcher = std::move(searcher);
}

template <typename T>
void
Orchestrator<T>::set_codec(CodecHostPtr<T> codec) noexcept {

    // Replace the current codec with the supplied one.
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

    // Append a solver to the execution sequence.
    _solvers.push_back(std::move(solver));
}

template <typename T>
const SpatialHashingSearcherHostPtr<T>&
Orchestrator<T>::searcher() const noexcept {

    return _searcher;
}

template <typename T>
const CodecHostPtr<T>&
Orchestrator<T>::codec() const noexcept {

    // Return the currently configured codec.
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

    // Return the ordered solver sequence.
    return _solvers;
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

    // Store the codec to be used by the constructed orchestrator.
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
Orchestrator<T>::Builder::with_solver(SolveHostPtr<T> solver) noexcept {

    // Append a solver to the builder's solver sequence.
    _solvers.push_back(std::move(solver));
    return *this;
}

template <typename T>
void
Orchestrator<T>::Builder::validate() const {

    // Every configured solver must be valid.
    for (const auto& solver : _solvers) {

        if (!solver) {
            throw std::runtime_error("Orchestrator::Builder: solver must not be null.");
        }
    }
}

template <typename T>
Orchestrator<T>
Orchestrator<T>::Builder::build() const {

    // Validate builder state before constructing the value object.
    validate();
    return Orchestrator<T>(_searcher, _codec, _measurer, _solvers);
}

template <typename T>
atlas::host_shared_ptr<Orchestrator<T>>
Orchestrator<T>::Builder::make_host_shared() const {

    // Validate builder state before constructing the shared object.
    validate();
    return atlas::make_host_shared<Orchestrator<T>>(_searcher, _codec, _measurer, _solvers);
}

} // namespace atlas::system
