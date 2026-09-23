/**
 * @file
 * @brief Defines a non-owning geometry view for rendering.
 */

#pragma once

#include "config/simulation_config.h"

#include <atlas/sync/sync.h>

namespace atlas::interactive {

/// Non-owning configured geometry plus its current world transform.
struct GeometryRenderView {
    /// Semantic boundary role used to select visualization styling.
    enum class Role {
        source,  ///< Particle-emission boundary.
        collider, ///< Particle-reflection boundary.
        sink     ///< Particle-removal boundary.
    };

    const SimulationConfig::Geometry* geometry = nullptr; ///< Borrowed geometry configuration.
    atlas::Sync sync; ///< Current local-to-world rigid transform.
    Role role = Role::collider; ///< Boundary role represented by this view.
};

}
