#pragma once

namespace atlas::detail {

template <typename T>
void
OrchestratorSearchStage<T>::run(Orchestrator<T>& orchestrator) const {
    if (orchestrator.searcher()) {
        orchestrator.searcher()->invalidate();
    }

    orchestrator.search();
}

template <typename T>
void
OrchestratorClassificationStage<T>::run(Orchestrator<T>& orchestrator) const {
    orchestrator.classify();
}

template <typename T>
void
OrchestratorMeasurementStage<T>::run(Orchestrator<T>& orchestrator, const T dt) const {
    orchestrator.measure(dt);
}

template <typename T>
void
OrchestratorForceStage<T>::run(Orchestrator<T>& orchestrator, const T dt) const {
    if (dt == T(0)) {
        return;
    }

    if (!orchestrator.make_probe()) {
        return;
    }

    orchestrator.apply_probe_forces(dt);
}

template <typename T>
void
OrchestratorSolveStage<T>::run(Orchestrator<T>& orchestrator, const T dt) const {
    orchestrator.solve(dt);
}

template <typename T>
void
OrchestratorPipeline<T>::run(Orchestrator<T>& orchestrator, const T dt) const {
    _search.run(orchestrator);
    _classification.run(orchestrator);
    _measurement.run(orchestrator, dt);
    _force.run(orchestrator, dt);
    _solve.run(orchestrator, dt);
}

}