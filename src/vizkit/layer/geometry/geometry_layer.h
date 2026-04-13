#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file geometry_layer.h
 * @brief Declares the base visualization layer for rendering geometry-driven line or primitive data in Vizkit.
 *
 * @details
 * This header defines @ref atlas::vizkit::GeometryLayer, an abstract base class
 * for visualization layers that render geometry derived from an optional
 * @ref atlas::Unit.
 *
 * The class bridges three responsibilities:
 * - generation of local-space geometry vertex data,
 * - synchronization of that data into world space using the unit transform,
 * - GPU-side upload and rendering through OpenGL objects and a shader program.
 *
 * ## Purpose
 * GeometryLayer exists to simplify the implementation of concrete visualization
 * layers such as:
 * - box layers,
 * - sphere or circle layers,
 * - cylinder layers,
 * - any other wireframe or primitive-based scene overlays.
 *
 * Instead of forcing every derived layer to manage OpenGL state directly, the
 * base class centralizes:
 * - VAO/VBO creation and destruction,
 * - shader program ownership,
 * - per-frame synchronization and draw preparation,
 * - MVP uniform setup through the active camera.
 *
 * ## Local and world space
 * Derived classes implement @ref build_geometry and generate vertices in
 * **local space**.
 *
 * The base class then typically:
 * - uses the bound unit's sync transform,
 * - converts those local vertices into **world space**,
 * - uploads the world-space positions for rendering.
 *
 * This separation makes it easy to reuse the same geometry-generation logic for
 * differently placed scene objects.
 *
 * ## Primitive mode
 * Rendering is parameterized by an OpenGL primitive mode such as:
 * - `GL_LINES`,
 * - `GL_LINE_STRIP`,
 * - `GL_POINTS`,
 * - or another primitive topology compatible with the generated geometry.
 *
 * The mode is stored in @ref _primitive_mode and is supplied during construction.
 *
 * ## Rendering lifecycle
 * A typical lifecycle for a geometry layer is:
 * 1. construct the layer and optionally bind a unit,
 * 2. call @ref init once with a valid GLFW window and camera,
 * 3. call @ref update each frame,
 * 4. call @ref shutdown to release GPU resources.
 *
 * ## Derived-class contract
 * Derived geometry layers must implement:
 * - @ref build_geometry, which produces local-space positions.
 *
 * They may optionally override:
 * - @ref synchronize, when custom local-to-world synchronization logic is needed.
 *
 * ## Ownership model
 * The class stores:
 * - an optional host-side shared pointer to a bound unit,
 * - a unique shader-program instance,
 * - CPU-side local/world position buffers,
 * - GPU-side VAO/VBO handles.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for geometry coordinates.
 */

#include <atlas/atlas.h>

#include <vizkit/layer/layer.h>
#include <vizkit/shader/shader_program.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

/**
 * @brief Abstract base class for visualization layers that render generated geometry.
 *
 * @details
 * @ref GeometryLayer provides a reusable rendering pipeline for derived classes
 * that generate vertex positions procedurally or from analytic scene objects.
 *
 * A geometry layer owns:
 * - an optional @ref atlas::Unit describing the source geometry and pose,
 * - an OpenGL vertex array object,
 * - an OpenGL vertex buffer object,
 * - a shader program used to render the geometry,
 * - CPU-side local and world position buffers.
 *
 * ## Responsibilities of the base class
 * The base class is responsible for:
 * - initializing OpenGL resources,
 * - requesting local-space geometry from the derived layer,
 * - synchronizing local geometry into world coordinates,
 * - uploading world-space vertex data,
 * - preparing shader uniforms such as MVP and color,
 * - shutting down owned GPU resources.
 *
 * ## Responsibilities of derived classes
 * Derived classes are responsible for:
 * - defining the local-space geometry through @ref build_geometry,
 * - optionally customizing synchronization behavior through @ref synchronize.
 *
 * ## Synchronization model
 * The default synchronization step typically:
 * - copies or transforms local vertex positions into world positions,
 * - uses the associated unit transform if present,
 * - optionally updates the unit over time using @p dt.
 *
 * The exact policy is implementation-defined in `geometry_layer.hpp`.
 *
 * ## Intended use
 * This base class is suitable for wireframe and simple primitive rendering of:
 * - analytic geometry,
 * - transformed unit instances,
 * - debugging overlays,
 * - helper visualizations in Vizkit.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for geometry coordinates.
 */
template <typename T>
class GeometryLayer : public Layer<T> {
public:
    /**
     * @brief Construct a geometry layer with a chosen primitive mode and optional unit.
     *
     * @details
     * The primitive mode controls how uploaded vertices are interpreted during
     * rendering. The optional unit provides:
     * - the source scene object,
     * - the local-to-world transform used during synchronization.
     *
     * @param primitive_mode OpenGL primitive mode used for rendering.
     * @param unit Optional bound unit used for synchronization.
     */
    ATLAS_HOST explicit GeometryLayer(
        unsigned int primitive_mode,
        const atlas::UnitHostPtr<T>& unit = nullptr);

    /**
     * @brief Virtual destructor.
     *
     * @details
     * Defaulted because resource-release behavior is handled explicitly through
     * @ref shutdown.
     */
    ~GeometryLayer() override = default;

    /**
     * @brief Initialize GPU resources and shader state for this layer.
     *
     * @details
     * This method is typically called once before the first render/update cycle.
     *
     * Common initialization tasks include:
     * - creating the VAO and VBO,
     * - constructing or compiling the shader program,
     * - resolving uniform locations,
     * - generating the initial local and/or world geometry buffers.
     *
     * @param window Active GLFW window.
     * @param camera Active visualization camera.
     */
    ATLAS_HOST void
    init(GLFWwindow* window, Camera& camera) override;

    /**
     * @brief Update and render-ready synchronize this layer for the current frame.
     *
     * @details
     * This function is typically called once per frame and may perform:
     * - local geometry generation if needed,
     * - local-to-world synchronization,
     * - unit-driven pose updates,
     * - CPU-to-GPU vertex upload,
     * - MVP uniform preparation using the active camera.
     *
     * The exact behavior is implementation-defined in `geometry_layer.hpp`.
     *
     * @param window Active GLFW window.
     * @param camera Active visualization camera.
     * @param dt Frame timestep.
     */
    ATLAS_HOST void
    update(GLFWwindow* window, Camera& camera, T dt) override;

    /**
     * @brief Release GPU resources owned by this layer.
     *
     * @details
     * Typical resource cleanup includes:
     * - deleting the VAO,
     * - deleting the VBO,
     * - releasing the shader program.
     *
     * After shutdown, the layer is no longer renderable until reinitialized.
     */
    ATLAS_HOST void
    shutdown() override;

protected:
    /**
     * @brief Build the local-space geometry for this layer.
     *
     * @details
     * Derived classes must populate @p positions with vertices expressed in the
     * local coordinate frame of the associated geometry.
     *
     * These local vertices are later transformed into world space by
     * @ref synchronize and then uploaded to the GPU.
     *
     * @param positions Output vector receiving local-space vertex positions.
     */
    virtual void
    build_geometry(std::vector<Vector3<T>>& positions)
        = 0;

    /**
     * @brief Synchronize local-space geometry into world-space geometry.
     *
     * @details
     * The default implementation typically:
     * - uses the associated unit, if present,
     * - transforms @ref _local_positions into @p world_positions,
     * - optionally updates the unit over the elapsed timestep.
     *
     * Derived classes may override this function when:
     * - custom synchronization behavior is needed,
     * - the world-space data is not a simple transformed copy of local data,
     * - additional animation or filtering should occur.
     *
     * @param world_positions Output vector receiving synchronized world-space positions.
     * @param dt Frame timestep.
     * @return `true` if synchronization produced valid renderable data; otherwise `false`.
     */
    virtual bool
    synchronize(std::vector<Vector3<T>>& world_positions, T dt);

protected:
    /**
     * @brief Optional scene unit associated with this layer.
     *
     * @details
     * When present, the unit provides:
     * - source geometry identity,
     * - local-to-world pose information,
     * - optional dynamic motion state.
     */
    atlas::UnitHostPtr<T> _unit;

    /**
     * @brief OpenGL primitive mode used to render the geometry.
     *
     * @details
     * Examples include `GL_LINES`, `GL_LINE_STRIP`, and `GL_POINTS`.
     */
    unsigned int _primitive_mode = GL_LINES;

    /**
     * @brief OpenGL vertex array object handle.
     *
     * @details
     * Encapsulates vertex attribute binding state for the rendered geometry.
     */
    GLuint _vao = 0;

    /**
     * @brief OpenGL vertex buffer object handle.
     *
     * @details
     * Stores uploaded world-space vertex positions on the GPU.
     */
    GLuint _vbo = 0;

    /**
     * @brief Number of vertices currently prepared for rendering.
     *
     * @details
     * This value typically reflects the size of @ref _world_positions after
     * synchronization and upload.
     */
    int _vertex_count = 0;

    /**
     * @brief Cached shader uniform location for the MVP matrix.
     */
    GLint _u_mvp = -1;

    /**
     * @brief Cached shader uniform location for the render color.
     */
    GLint _u_color = -1;

    /**
     * @brief Owned shader program used to render the geometry layer.
     *
     * @details
     * The concrete shader content and lifecycle are implementation-defined in
     * `geometry_layer.hpp`.
     */
    std::unique_ptr<ShaderProgram> _program;

    /**
     * @brief CPU-side local-space vertex positions generated by @ref build_geometry.
     *
     * @details
     * These positions represent the geometry before application of any world
     * transform.
     */
    std::vector<Vector3<T>> _local_positions;

    /**
     * @brief CPU-side world-space vertex positions used for rendering.
     *
     * @details
     * These positions are typically derived from @ref _local_positions through
     * @ref synchronize and then uploaded to the GPU.
     */
    std::vector<Vector3<T>> _world_positions;
};

} // namespace atlas::vizkit

#include <vizkit/layer/geometry/geometry_layer.hpp>

#endif