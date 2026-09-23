/**
 * @file
 * @brief Declares rendering of configured simulation geometry.
 */

#pragma once

#include "rendering/layer/layer.h"
#include "rendering/opengl/buffer.h"
#include "rendering/opengl/shader.h"
#include "view/geometry_render_view.h"

#include <atlas/math/vector/float3.h>

#include <vector>

namespace atlas::interactive {

/// Tessellates configured boundaries and draws them as line segments.
class GeometryLayer final : public Layer {
public:
    ~GeometryLayer() override;

    /**
     * @brief Generates world-space line vertices for one geometry view.
     * @param view Geometry description, pose, and semantic role.
     * @param domain_lower Lower simulation-domain corner used for unbounded shapes.
     * @param domain_upper Upper simulation-domain corner used for unbounded shapes.
     */
    static std::vector<atlas::Float3>
    make_line_vertices(const GeometryRenderView& view,
                       const atlas::Float3& domain_lower,
                       const atlas::Float3& domain_upper);

    /// @copydoc Layer::initialize
    void initialize() override;
    /// @copydoc Layer::render
    void render(const RenderState& state,
                const Camera& camera,
                float aspect_ratio) override;
    /// @copydoc Layer::shutdown
    void shutdown() override;

private:
    std::vector<atlas::Float3> _vertices; ///< Reused CPU tessellation scratch buffer.
    opengl::Buffer _buffer; ///< Reused line-vertex buffer.
    opengl::Shader _shader; ///< Uniform-color line shader.
    unsigned int _vao = 0; ///< Vertex-array object describing line positions.
    int _view_projection_location = -1; ///< Camera-transform uniform location.
    int _color_location = -1; ///< Role-color uniform location.
};

}
