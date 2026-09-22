#pragma once

#include "config/simulation_config.h"

#include <atlas/sync/sync.h>

namespace atlas::interactive {

struct GeometryRenderView {
    enum class Role { source, collider, sink };

    const SimulationConfig::Geometry* geometry = nullptr;
    atlas::Sync sync;
    Role role = Role::collider;
};

}
