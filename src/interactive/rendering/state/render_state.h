#pragma once

#include "rendering/opengl/buffer.h"
#include "view/geometry_render_view.h"

#include <atlas/math/vector/float3.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace atlas::interactive {

struct RenderState {
    std::size_t particle_count = 0;
    opengl::Buffer position;
    opengl::Buffer velocity;
    opengl::Buffer species;
    std::optional<opengl::Buffer> temperature;
    std::optional<opengl::Buffer> translational_energy;
    std::optional<opengl::Buffer> rotational_energy;
    std::optional<opengl::Buffer> vibrational_energy;
    std::vector<GeometryRenderView> geometries;
    atlas::Float3 lower_corner;
    atlas::Float3 upper_corner;
};

}
