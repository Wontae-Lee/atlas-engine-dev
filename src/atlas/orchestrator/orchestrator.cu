#include <atlas/orchestrator/orchestrator.h>

#include <atlas/universe/universe_state.h>

#include <stdexcept>
#include <utility>

namespace atlas {

namespace {

    // Emplaces the state when the universe lacks it, resizes it when the grid
    // has changed under it.
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

Orchestrator::Orchestrator(FluidHostPtr fluid,
                           UniverseHostPtr universe,
                           HostBuffer<SolverHostPtr> solvers)
    : _fluid(std::move(fluid))
    , _universe(std::move(universe))
    , _solvers(std::move(solvers)) {

    initialize_states();
}

Orchestrator::Builder
Orchestrator::builder() noexcept {
    return Builder {};
}

void
Orchestrator::initialize_states() {
    if (!_universe) {
        return;
    }

    const auto cell_count = static_cast<std::size_t>(_universe->cell_count());

    // Whichever solvers run, the codec buckets each cell into one of them.
    ensure_universe_state<UniverseAllocatedSolverState>(*_universe, cell_count);

    for (const auto& solver : _solvers) {
        switch (solver->type()) {
            case SolverType::dsmc:
                initialize_dsmc_states();
                break;
        }
    }
}

void
Orchestrator::initialize_dsmc_states() {
    const auto cell_count = static_cast<std::size_t>(_universe->cell_count());

    // Cell occupancy comes from the searcher; the rest is the NTC bookkeeping
    // the collision step reads and writes.
    ensure_universe_state<UniverseNumberParticleState>(*_universe, cell_count);
    ensure_universe_state<UniverseMaxRelativeSpeedState>(*_universe, cell_count);
    ensure_universe_state<UniverseMaxSigmaGState>(*_universe, cell_count);
    ensure_universe_state<UniverseCollisionCountState>(*_universe, cell_count);
}

void
Orchestrator::orchestrate(const SpatialHashingSearcherView& searcher_view, const float dt) {
    if (!_fluid || !_universe) {
        return;
    }

    // A solver's position in the list is the index the codec allocates cells to,
    // so each one only touches the cells holding its own index.
    for (std::size_t i = 0; i < _solvers.size(); ++i) {
        _solvers[i]->solve(*_fluid, *_universe, searcher_view, static_cast<int>(i), dt);
    }
}

Orchestrator::Builder&
Orchestrator::Builder::with_fluid(FluidHostPtr fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

Orchestrator::Builder&
Orchestrator::Builder::with_universe(UniverseHostPtr universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

Orchestrator::Builder&
Orchestrator::Builder::with_solver(SolverHostPtr solver) {
    _solvers.push_back(std::move(solver));
    return *this;
}

void
Orchestrator::Builder::validate() const {
    if (!_fluid) {
        throw std::runtime_error("Orchestrator::Builder: fluid must not be null.");
    }

    if (!_universe) {
        throw std::runtime_error("Orchestrator::Builder: universe must not be null.");
    }

    if (_solvers.empty()) {
        throw std::runtime_error("Orchestrator::Builder: at least one solver is required.");
    }

    for (const auto& solver : _solvers) {
        if (!solver) {
            throw std::runtime_error("Orchestrator::Builder: a solver must not be null.");
        }
    }
}

Orchestrator
Orchestrator::Builder::build() const {
    validate();

    return Orchestrator(_fluid, _universe, _solvers);
}

atlas::host_shared_ptr<Orchestrator>
Orchestrator::Builder::make_host_shared() const {
    return atlas::make_host_shared<Orchestrator>(build());
}

}
