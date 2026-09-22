#pragma once

#include "view/geometry_render_view.h"
#include "view/simulation_render_view.h"

#include <atlas/math/vector/float3.h>

#include <vector>

namespace atlas::interactive {

struct SimulationSceneView {
    SimulationRenderView particles;
    std::vector<GeometryRenderView> geometries;
    atlas::Float3 lower_corner;
    atlas::Float3 upper_corner;
};

}
