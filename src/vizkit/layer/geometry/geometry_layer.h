#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/atlas.h>

#include <vizkit/layer/layer.h>
#include <vizkit/shader/shader_program.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

    template <typename T>
    class GeometryLayer : public Layer<T> {
    public:
        ATLAS_HOST explicit GeometryLayer(
            unsigned int primitive_mode,
            const atlas::UnitHostPtr<T>& unit = nullptr);

        ~GeometryLayer() override = default;

        ATLAS_HOST void
        init(GLFWwindow* window, Camera& camera) override;

        ATLAS_HOST void
        update(GLFWwindow* window, Camera& camera, T dt) override;

        ATLAS_HOST void
        shutdown() override;

    protected:
        virtual void
        build_geometry(std::vector<Vector3<T>>& positions) = 0;

        virtual bool
        synchronize(std::vector<Vector3<T>>& world_positions, T dt);

    protected:
        atlas::UnitHostPtr<T> _unit;

        unsigned int _primitive_mode = GL_LINES;

        GLuint _vao = 0;
        GLuint _vbo = 0;

        int _vertex_count = 0;
        GLint _u_mvp      = -1;
        GLint _u_color    = -1;

        std::unique_ptr<ShaderProgram> _program;

        std::vector<Vector3<T>> _local_positions;
        std::vector<Vector3<T>> _world_positions;
    };

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/geometry_layer.hpp>

#endif