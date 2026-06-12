#pragma once

#include <atlas/core/macros.h>
#include <atlas/orchestrator/orchestrator_probe.h>
#include <atlas/parallel/parallel_for.h>

namespace atlas::detail {

/**
 * @brief Applies force-related universe states through an orchestration probe.
 */
template <typename T>
class OrchestratorForceApplier final {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_all(const OrchestratorProbe<T>& probe, T dt) const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_gravity(const OrchestratorProbe<T>& probe, T dt) const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply_field_force(const OrchestratorProbe<T>& probe, T dt) const;

private:
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_gravity(const OrchestratorProbe<T>& probe) const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_field_force(const OrchestratorProbe<T>& probe) const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    force_cell_count(const OrchestratorProbe<T>& probe, bool gravity, bool field_force) const noexcept;
};

} // namespace atlas::detail

#include <atlas/orchestrator/detail/orchestrator_force_applier.hpp>
