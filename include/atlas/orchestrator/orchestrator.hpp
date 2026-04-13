#pragma once

namespace atlas::system {

template <typename T>
Orchestrator<T>::Orchestrator(SolveHostPtr<T> solver) noexcept
    : _solver(std::move(solver)) {
    // Construct the orchestrator with an optional solver backend.
    //
    // The orchestrator itself does not implement numerical solving logic.
    // Instead, it acts as a thin forwarding layer that owns or references
    // the concrete solver object through _solver.
    //
    // If solver is null:
    // - the orchestrator remains in a valid but inactive state
    // - solve() will simply return without doing any work
}

template <typename T>
typename Orchestrator<T>::Builder
Orchestrator<T>::builder() noexcept {
    // Return a fresh builder for staged Orchestrator<T> construction.
    return Builder {};
}

template <typename T>
void
Orchestrator<T>::solve(DomainDeviceProbe<T>& domain,
                       SpatialHashingProbe<T>& searcher,
                       FluidDeviceProbe<T>& particle,
                       CodecDeviceProbe<T>& codec) {
    // Dispatch the solve step to the configured concrete solver.
    //
    // Inputs:
    // - domain   : device-visible domain state
    // - searcher : device-visible spatial neighborhood / hashing structure
    // - particle : device-visible particle state
    // - codec    : device-visible codec state used by the current simulation pipeline
    //
    // The orchestrator does not transform or reinterpret these arguments.
    // Its role is simply to centralize solver invocation.

    // If no solver is configured, solving is skipped.
    if (!_solver) {
        return;
    }

    // Forward the solve request to the bound solver implementation.
    _solver->solve(domain, searcher, particle, codec);
}

template <typename T>
void
Orchestrator<T>::set_solver(SolveHostPtr<T> solver) noexcept {
    // Replace the currently configured solver.
    //
    // This allows the orchestration layer to switch solver implementations
    // at runtime without changing the external call site.
    _solver = std::move(solver);
}

template <typename T>
const SolveHostPtr<T>&
Orchestrator<T>::solver() const noexcept {
    // Return the currently configured solver object.
    //
    // The caller receives a const reference to the owning/shared pointer,
    // not a copied solver instance.
    return _solver;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_solver(SolveHostPtr<T> solver) noexcept {
    // Stage the solver dependency inside the builder.
    //
    // The solver is moved into the builder so build() can later materialize
    // the final Orchestrator<T> object from a validated configuration.
    _solver = std::move(solver);
    return *this;
}

template <typename T>
void
Orchestrator<T>::Builder::validate() const {
    // Validate that the builder has all required inputs.
    //
    // Current rule:
    // - a concrete solver must be provided before an orchestrator can be built
    if (!_solver) {
        throw std::runtime_error("Orchestrator::Builder: solver must not be null.");
    }
}

template <typename T>
Orchestrator<T>
Orchestrator<T>::Builder::build() const {
    // Build an Orchestrator<T> by value after validation succeeds.
    validate();
    return Orchestrator<T>(_solver);
}

template <typename T>
atlas::host_shared_ptr<Orchestrator<T>>
Orchestrator<T>::Builder::make_host_shared() const {
    // Build an Orchestrator<T> in host-shared storage after validation succeeds.
    //
    // This is the shared-pointer convenience counterpart to build().
    validate();
    return atlas::make_host_shared<Orchestrator<T>>(_solver);
}

} // namespace atlas::system