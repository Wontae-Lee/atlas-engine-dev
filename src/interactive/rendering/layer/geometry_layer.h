#pragma once

#include "rendering/layer/layer.h"
#include "rendering/opengl/buffer.h"
#include "rendering/opengl/shader.h"
#include "view/geometry_render_view.h"

#include <atlas/math/vector/float3.h>

#include <vector>

namespace atlas::interactive {

class GeometryLayer final : public Layer {
public:
    ~GeometryLayer() override;

    static std::vector<atlas::Float3>
    make_line_vertices(const GeometryRenderView& view,
                       const atlas::Float3& domain_lower,
                       const atlas::Float3& domain_upper);

    void initialize() override;
    void render(const RenderState& state,
                const Camera& camera,
                float aspect_ratio) override;
    void shutdown() override;

private:
    std::vector<atlas::Float3> _vertices;
    opengl::Buffer _buffer;
    opengl::Shader _shader;
    unsigned int _vao = 0;
    int _view_projection_location = -1;
    int _color_location = -1;
};

}
