#pragma once

#include <atlas/core/macros.h>

namespace atlas {

template <typename T>
class Orchestrator;

}

namespace atlas::detail {

template <typename T>
struct OrchestratorSearchStage final {
    ATLAS_HOST ATLAS_FORCE_INLINE void
    run(Orchestrator<T>& orchestrator) const;
};

template <typename T>
struct OrchestratorClassificationStage final {
    ATLAS_HOST ATLAS_FORCE_INLINE void
    run(Orchestrator<T>& orchestrator) const;
};

template <typename T>
struct OrchestratorMeasurementStage final {
    ATLAS_HOST ATLAS_FORCE_INLINE void
    run(Orchestrator<T>& orchestrator, T dt) const;
};

template <typename T>
struct OrchestratorForceStage final {
    ATLAS_HOST ATLAS_FORCE_INLINE void
    run(Orchestrator<T>& orchestrator, T dt) const;
};

template <typename T>
struct OrchestratorSolveStage final {
    ATLAS_HOST ATLAS_FORCE_INLINE void
    run(Orchestrator<T>& orchestrator, T dt) const;
};

template <typename T>
class OrchestratorPipeline final {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    run(Orchestrator<T>& orchestrator, T dt) const;

private:
    OrchestratorSearchStage<T> _search {};
    OrchestratorClassificationStage<T> _classification {};
    OrchestratorMeasurementStage<T> _measurement {};
    OrchestratorForceStage<T> _force {};
    OrchestratorSolveStage<T> _solve {};
};

}

#include <atlas/orchestrator/detail/orchestrator_pipeline.hpp>