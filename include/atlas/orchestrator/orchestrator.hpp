#pragma once

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept>

namespace atlas::system {

template <typename T>
Orchestrator<T>::Orchestrator(CodecHostPtr<T> codec,
                              HostBuffer<SolveHostPtr<T>> solvers) noexcept
    : _codec(std::move(codec))
    , _solvers(std::move(solvers)) {
}

template <typename T>
typename Orchestrator<T>::Builder
Orchestrator<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
void
Orchestrator<T>::orchestrate() {

    if (_solvers.empty()) {
        return;
    }

    if (!_codec) {

        for (const auto& solver : _solvers) {

            if (!solver) continue;

            solver->solve();
        }

        return;
    }

    const auto* allocated_solver = atlas::raw_pointer_cast(_codec->allocated_solver().data());
    const auto solver_count = static_cast<int>(_solvers.size());

    for (int i = 0; i < solver_count; ++i) {

        const auto& solver = _solvers[static_cast<std::size_t>(i)];

        if (!solver) continue;

        solver->solve(allocated_solver[i]);
    }
}

template <typename T>
void
Orchestrator<T>::set_codec(CodecHostPtr<T> codec) noexcept {

    _codec = std::move(codec);
}

template <typename T>
void
Orchestrator<T>::add_solver(SolveHostPtr<T> solver) noexcept {

    _solvers.push_back(std::move(solver));
}

template <typename T>
const CodecHostPtr<T>&
Orchestrator<T>::codec() const noexcept {

    return _codec;
}

template <typename T>
const HostBuffer<SolveHostPtr<T>>&
Orchestrator<T>::solvers() const noexcept {

    return _solvers;
}

template <typename T>
typename Orchestrator<T>::Builder&
Orchestrator<T>::Builder::with_codec(CodecHostPtr<T> codec) noexcept {

    _codec = std::move(codec);
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
Orchestrator<T>
Orchestrator<T>::Builder::build() const {

    validate();
    return Orchestrator<T>(_codec, _solvers);
}

template <typename T>
atlas::host_shared_ptr<Orchestrator<T>>
Orchestrator<T>::Builder::make_host_shared() const {

    validate();
    return atlas::make_host_shared<Orchestrator<T>>(_codec, _solvers);
}

}
