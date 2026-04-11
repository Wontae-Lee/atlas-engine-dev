#pragma once

namespace atlas::system {

template <typename T>
Orchestrator<T>::Orchestrator(SolveHostPtr<T> solver) noexcept
    : _solver(std::move(solver)) { }

template <typename T>
typename Orchestrator<T>::Builder
Orchestrator<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Orchestrator<T>::solve(DomainDeviceProbe<T> domain,
                       SpatialHashingProbe<T> searcher,
                       FluidDeviceProbe<T> particle,
                       CodecDeviceProbe<T> codec) {
    if (!_solver) {
        return;
    }

    _solver->solve(domain, searcher, particle, codec);
}

template <typename T>
void
Orchestrator<T>::set_solver(SolveHostPtr<T> solver) noexcept {
    _solver = std::move(solver);
}

template <typename T>
const SolveHostPtr<T>&
Orchestrator<T>::solver() const noexcept {
    return _solver;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_solver(SolveHostPtr<T> solver) noexcept {
    _solver = std::move(solver);
    return *this;
}

template <typename T>
void
Orchestrator<T>::Builder::validate() const {
    if (!_solver) {
        throw std::runtime_error("Orchestrator::Builder: solver must not be null.");
    }
}

template <typename T>
Orchestrator<T>
Orchestrator<T>::Builder::build() const {
    validate();
    return Orchestrator<T>(_solver);
}

template <typename T>
atlas::host_shared_ptr<Orchestrator<T>>
Orchestrator<T>::Builder::make_host_shared() const {
    validate();
    return atlas::make_host_shared<Orchestrator<T>>(_solver);
}

}