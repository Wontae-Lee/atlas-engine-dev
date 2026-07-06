#pragma once

#include <atlas/core/macros.h>
#include <atlas/orchestrator/orchestrator_probe.h>

namespace atlas::detail {

class OrchestratorForceApplier final {
public:
    ATLAS_HOST void
    apply_all(const OrchestratorProbe& probe, float dt) const;

    ATLAS_HOST void
    apply_gravity(const OrchestratorProbe& probe, float dt) const;

    ATLAS_HOST void
    apply_field_force(const OrchestratorProbe& probe, float dt) const;

private:
    ATLAS_HOST ATLAS_NODISCARD bool
    has_gravity(const OrchestratorProbe& probe) const noexcept;

    ATLAS_HOST ATLAS_NODISCARD bool
    has_field_force(const OrchestratorProbe& probe) const noexcept;

    ATLAS_HOST ATLAS_NODISCARD int
    force_cell_count(const OrchestratorProbe& probe, bool gravity, bool field_force) const noexcept;
};

}
