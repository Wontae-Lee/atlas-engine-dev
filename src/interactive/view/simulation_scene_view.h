/**
 * @file
 * @brief Defines the complete non-owning scene consumed by rendering.
 */

#pragma once

#include "view/geometry_render_view.h"
#include "view/simulation_render_view.h"

#include <atlas/math/vector/float3.h>

#include <vector>

namespace atlas::interactive {

/// Complete non-owning scene description consumed by StateProvider.
struct SimulationSceneView {
    SimulationRenderView particles; ///< Live particle buffer views.
    std::vector<GeometryRenderView> geometries; ///< Configured boundary views.
    atlas::Float3 lower_corner; ///< Lower simulation-domain corner.
    atlas::Float3 upper_corner; ///< Upper simulation-domain corner.
};

}
