#pragma once

#include <atlas/core/macros.h>

namespace atlas {

class Orchestrator;

}

namespace atlas::detail {

struct OrchestratorSearchStage final {
    ATLAS_HOST void
    run(Orchestrator& orchestrator) const;
};

struct OrchestratorClassificationStage final {
    ATLAS_HOST void
    run(Orchestrator& orchestrator) const;
};

struct OrchestratorMeasurementStage final {
    ATLAS_HOST void
    run(Orchestrator& orchestrator, float dt) const;
};

struct OrchestratorForceStage final {
    ATLAS_HOST void
    run(Orchestrator& orchestrator, float dt) const;
};

struct OrchestratorSolveStage final {
    ATLAS_HOST void
    run(Orchestrator& orchestrator, float dt) const;
};

class OrchestratorPipeline final {
public:
    ATLAS_HOST void
    run(Orchestrator& orchestrator, float dt) const;

private:
    OrchestratorSearchStage _search {};
    OrchestratorClassificationStage _classification {};
    OrchestratorMeasurementStage _measurement {};
    OrchestratorForceStage _force {};
    OrchestratorSolveStage _solve {};
};

}
