/**
 * @file
 * @brief Defines graphics resources representing one renderable simulation state.
 */

#pragma once

#include "rendering/opengl/buffer.h"
#include "view/geometry_render_view.h"

#include <atlas/math/vector/float3.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace atlas::interactive {

/**
 * @brief Graphics-side snapshot consumed by rendering layers.
 *
 * Buffers contain only the live particle prefix. Optional buffers exist only
 * when the corresponding state exists in the Atlas Fluid.
 */
struct RenderState {
    std::size_t particle_count = 0; ///< Number of live particles represented by every state.
    opengl::Buffer position; ///< Packed atlas::Float3 positions.
    opengl::Buffer velocity; ///< Packed atlas::Float3 velocities.
    opengl::Buffer species; ///< Packed material indices.
    std::optional<opengl::Buffer> temperature; ///< Optional scalar temperature state.
    std::optional<opengl::Buffer> translational_energy; ///< Optional scalar translational energy.
    std::optional<opengl::Buffer> rotational_energy; ///< Optional scalar rotational energy.
    std::optional<opengl::Buffer> vibrational_energy; ///< Optional scalar vibrational energy.
    std::vector<GeometryRenderView> geometries; ///< Non-owning configured geometry views.
    atlas::Float3 lower_corner; ///< Lower simulation-domain corner.
    atlas::Float3 upper_corner; ///< Upper simulation-domain corner.
};

}
