#pragma once

#include <atlas/core/macros.h>

namespace atlas {

template <typename T>
class Orchestrator;

} // namespace atlas

namespace atlas::detail {

/**
 * @brief Invalidates and rebuilds spatial search data.
 */
template <typename T>
struct OrchestratorSearchStage final {
    ATLAS_HOST ATLAS_FORCE_INLINE void
    run(Orchestrator<T>& orchestrator) const;
};

/**
 * @brief Updates solver classification or allocation data.
 */
template <typename T>
struct OrchestratorClassificationStage final {
    ATLAS_HOST ATLAS_FORCE_INLINE void
    run(Orchestrator<T>& orchestrator) const;
};

/**
 * @brief Executes simulation measurement for the current time step.
 */
template <typename T>
struct OrchestratorMeasurementStage final {
    ATLAS_HOST ATLAS_FORCE_INLINE void
    run(Orchestrator<T>& orchestrator, T dt) const;
};

/**
 * @brief Applies cell gravity and field-force states.
 */
template <typename T>
struct OrchestratorForceStage final {
    ATLAS_HOST ATLAS_FORCE_INLINE void
    run(Orchestrator<T>& orchestrator, T dt) const;
};

/**
 * @brief Dispatches configured solvers.
 */
template <typename T>
struct OrchestratorSolveStage final {
    ATLAS_HOST ATLAS_FORCE_INLINE void
    run(Orchestrator<T>& orchestrator, T dt) const;
};

/**
 * @brief Names the fixed solver-side orchestration order.
 */
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

} // namespace atlas::detail

#include <atlas/orchestrator/detail/orchestrator_pipeline.hpp>
