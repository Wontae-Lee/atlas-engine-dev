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

/**
 * @brief Base class for renderable geometry layers backed by vertex positions.
 *
 * @tparam T Scalar type used for geometry coordinates and time-step values.
 *
 * `GeometryLayer` implements the common VizKit pipeline for simple geometry:
 * 1. A derived class generates local-space vertex positions via
 *    @ref build_geometry.
 * 2. The layer optionally transforms those positions into world space using the
 *    bound unit's @ref atlas::SyncOperator.
 * 3. The transformed vertices are uploaded to an OpenGL vertex buffer.
 * 4. Each frame, the layer updates the unit, refreshes synchronized vertices
 *    if needed, builds an MVP matrix from the active camera, and issues a draw.
 *
 * This class is intentionally position-only: it manages one vertex buffer of
 * `Vector3<T>` coordinates and renders them with a fixed shader program. Shape-
 * specific topology generation is delegated to subclasses.
 *
 * Responsibilities of derived classes:
 * - define the local-space geometry in @ref build_geometry
 * - optionally override @ref synchronize if world-space mapping requires custom
 *   behavior beyond point-wise sync through the unit transform
 *
 * Responsibilities handled by this base class:
 * - OpenGL VAO/VBO lifetime management
 * - shader program creation and uniform lookup
 * - unit update dispatch during rendering
 * - camera MVP upload
 * - issuing `glDrawArrays` with the configured primitive mode
 *
 * @note If no unit is bound, synchronization degenerates to copying local
 * positions directly into the world-space buffer.
 */
template <typename T>
class GeometryLayer : public Layer<T> {
public:
    /**
     * @brief Constructs a geometry layer with a draw mode and optional unit.
     *
     * @param primitive_mode OpenGL primitive type passed to `glDrawArrays`,
     * such as `GL_LINES` or `GL_TRIANGLES`.
     * @param unit Optional unit that provides transform and geometry context.
     */
    ATLAS_HOST explicit GeometryLayer(
        unsigned int primitive_mode,
        const atlas::UnitHostPtr<T>& unit = nullptr);

    ~GeometryLayer() override = default;

    /**
     * @brief Initializes geometry, GPU resources, and shader state.
     *
     * @param window Active GLFW window.
     * @param camera Active camera used by the layer interface.
     *
     * Initialization workflow:
     * 1. Clear and rebuild local geometry through @ref build_geometry.
     * 2. Initialize the world-space vertex buffer from local positions.
     * 3. Perform one synchronization pass.
     * 4. Create the shader program and OpenGL VAO/VBO if there are vertices.
     * 5. Upload the initial world-space positions and cache uniform locations.
     *
     * If the derived geometry generator produces no vertices, the method exits
     * early and no GPU resources are created.
     */
    ATLAS_HOST void
    init(GLFWwindow* window, Camera& camera) override;

    /**
     * @brief Updates the bound unit, refreshes synchronized vertices, and draws.
     *
     * @param window Active GLFW window used to query framebuffer size.
     * @param camera Active camera used to build the MVP matrix.
     * @param dt Frame delta time passed through to the unit and synchronization.
     *
     * Per-frame behavior:
     * - returns immediately if initialization did not produce drawable state
     * - updates the bound unit, if present
     * - calls @ref synchronize and updates the VBO only when positions changed
     * - builds the current model-view-projection matrix from the camera
     * - binds shader state and draws the vertex array with `_primitive_mode`
     */
    ATLAS_HOST void
    update(GLFWwindow* window, Camera& camera, T dt) override;

    /**
     * @brief Releases OpenGL resources and resets cached state.
     *
     * Deletes the VBO and VAO if they exist, clears CPU-side position buffers,
     * resets uniform handles, and destroys the shader program.
     */
    ATLAS_HOST void
    shutdown() override;

protected:
    /**
     * @brief Builds the layer's local-space geometry.
     *
     * @param positions Output vector filled with local-space vertex positions.
     *
     * Derived classes must populate `positions` according to the configured
     * primitive topology. The result is later transformed into world space and
     * rendered with `glDrawArrays`.
     */
    virtual void
    build_geometry(std::vector<Vector3<T>>& positions)
        = 0;

    /**
     * @brief Synchronizes local vertices into world-space render positions.
     *
     * @param world_positions Buffer to be updated with synchronized positions.
     * @param dt Frame delta time.
     * @return `true` if any synchronized position changed enough to require a
     * GPU buffer update; otherwise `false`.
     *
     * The default implementation performs point-wise transformation:
     * - if no unit is bound, copy local positions directly
     * - otherwise, transform each local vertex through the unit's
     *   @ref atlas::SyncOperator using point semantics
     *
     * A small epsilon comparison is used to detect whether the world-space
     * positions changed relative to the previously stored buffer.
     *
     * Derived classes may override this when synchronization depends on dynamic
     * geometry regeneration or on a mapping other than simple point transforms.
     */
    virtual bool
    synchronize(std::vector<Vector3<T>>& world_positions, T dt);

protected:
    /** @brief Optional unit supplying transforms and geometry context. */
    atlas::UnitHostPtr<T> _unit;

    /** @brief OpenGL primitive mode used when issuing `glDrawArrays`. */
    unsigned int _primitive_mode = GL_LINES;

    /** @brief Vertex array object describing the bound vertex layout. */
    GLuint _vao = 0;
    /** @brief Vertex buffer object storing world-space positions. */
    GLuint _vbo = 0;

    /** @brief Number of vertices currently uploaded and drawable. */
    int _vertex_count = 0;
    /** @brief Cached shader uniform location for the MVP matrix. */
    GLint _u_mvp      = -1;
    /** @brief Cached shader uniform location for the draw color. */
    GLint _u_color    = -1;

    /** @brief Shader program used to render the geometry layer. */
    std::unique_ptr<ShaderProgram> _program;

    /** @brief Geometry generated by the derived class in local coordinates. */
    std::vector<Vector3<T>> _local_positions;
    /** @brief Render-ready positions after synchronization into world space. */
    std::vector<Vector3<T>> _world_positions;
};

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/geometry_layer.hpp>

#endif
