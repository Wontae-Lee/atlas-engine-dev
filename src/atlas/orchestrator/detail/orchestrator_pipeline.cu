#include <atlas/orchestrator/detail/orchestrator_pipeline.h>

#include <atlas/orchestrator/orchestrator.h>

namespace atlas::detail {

void
OrchestratorSearchStage::run(Orchestrator& orchestrator) const {
    if (orchestrator.searcher()) {
        orchestrator.searcher()->invalidate();
    }

    orchestrator.search();
}

void
OrchestratorClassificationStage::run(Orchestrator& orchestrator) const {
    orchestrator.classify();
}

void
OrchestratorMeasurementStage::run(Orchestrator& orchestrator, const float dt) const {
    orchestrator.measure(dt);
}

void
OrchestratorForceStage::run(Orchestrator& orchestrator, const float dt) const {
    if (dt == 0.0f) {
        return;
    }

    if (!orchestrator.make_probe()) {
        return;
    }

    orchestrator.apply_probe_forces(dt);
}

void
OrchestratorSolveStage::run(Orchestrator& orchestrator, const float dt) const {
    orchestrator.solve(dt);
}

void
OrchestratorPipeline::run(Orchestrator& orchestrator, const float dt) const {
    _search.run(orchestrator);
    _classification.run(orchestrator);
    _measurement.run(orchestrator, dt);
    _force.run(orchestrator, dt);
    _solve.run(orchestrator, dt);
}

}
