#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file particle_layer.h
 * @brief Declares a visualization layer that renders simulation particles from an Atlas system.
 *
 * @details
 * This header defines @ref atlas::vizkit::ParticleLayer, a concrete
 * @ref atlas::vizkit::Layer implementation responsible for visualizing the
 * currently active particle positions stored in an @ref atlas::System.
 *
 * ## Purpose
 * The particle layer is intended for:
 * - rendering live simulation particles,
 * - inspecting particle motion and spatial distribution,
 * - debugging source, sink, solver, and collider behavior,
 * - providing a simple real-time point-based representation of simulation state.
 *
 * ## Data source
 * The layer reads particle positions from the bound @ref atlas::System, which
 * owns the canonical runtime particle buffers for the simulation.
 *
 * The particle layer does not generate synthetic geometry. Instead, it:
 * - accesses the active particle positions from the system,
 * - uploads or maps those positions into an OpenGL vertex buffer,
 * - renders them as a point cloud.
 *
 * ## Rendering model
 * The layer uses:
 * - one OpenGL vertex array object,
 * - one OpenGL vertex buffer object,
 * - one shader program,
 * - one configurable RGBA particle color.
 *
 * The number of rendered particles is controlled by the current active particle
 * count in the bound system.
 *
 * ## Backend behavior
 * The upload path depends on the active build configuration:
 *
 * - When `ATLAS_TASKING_CUDA` is enabled:
 *   - the VBO may be registered with CUDA through a graphics interop resource,
 *   - particle positions may be transferred or mapped through CUDA/OpenGL
 *     interoperability.
 *
 * - Otherwise:
 *   - particle positions are staged in a host-side vector,
 *   - the host copy is uploaded into the VBO through the non-CUDA path.
 *
 * ## Lifecycle
 * A typical layer lifecycle is:
 * 1. construct the layer with a valid system,
 * 2. call @ref init once,
 * 3. call @ref update each frame,
 * 4. call @ref shutdown when rendering is complete.
 *
 * ## Color
 * Particle rendering color is stored as a 4D vector:
 * - RGB components define the particle color,
 * - A defines alpha/transparency.
 *
 * ## Builder support
 * The nested @ref Builder provides a fluent interface to:
 * - assign the target system,
 * - configure the particle color,
 * - construct the layer by value or shared ownership.
 *
 * ---
 */

#include <atlas/core/macros.h>
#include <atlas/math/vector/vector4.h>
#include <atlas/system/system.h>

#include <vizkit/layer/layer.h>
#include <vizkit/shader/shader_program.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

/**
 * @brief Visualization layer that renders active particles from an Atlas system.
 *
 * @details
 * @ref ParticleLayer is a point-based rendering layer that visualizes the active
 * particle positions contained in a bound @ref atlas::System.
 *
 * The layer is responsible for:
 * - validating that a usable system is available,
 * - initializing GPU-side rendering resources,
 * - reading particle positions from the runtime system,
 * - uploading or mapping those positions into a GPU vertex buffer,
 * - rendering the resulting particle cloud each frame.
 *
 * ## System dependency
 * The bound system is expected to provide:
 * - a valid fluid subsystem,
 * - a valid particle/device probe,
 * - particle positions stored in a buffer accessible through the configured
 *   backend.
 *
 * ## Rendering characteristics
 * The layer typically renders particles as:
 * - point primitives,
 * - uniformly colored instances,
 * - one vertex per active particle position.
 *
 * ## Capacity tracking
 * The layer stores an internal capacity value used to track the allocated size
 * of the visualization buffer. This allows the implementation to resize the VBO
 * only when necessary instead of every frame.
 *
 * ## Non-CUDA path
 * In non-CUDA builds, particle positions are copied into the host-side vector
 * @ref _host_positions before being uploaded into the GPU buffer.
 *
 * ## CUDA path
 * In CUDA-enabled builds, @ref _cuda_vbo_resource may be used to register the
 * VBO for CUDA/OpenGL interop, allowing more direct access to the buffer.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for particle coordinates and color.
 */
template <typename T>
class ParticleLayer final : public Layer<T> {
public:
    /**
     * @brief Fluent builder for configuring and constructing @ref ParticleLayer.
     *
     * @details
     * The builder stages:
     * - the target system,
     * - the particle render color,
     * and constructs either:
     * - a layer by value, or
     * - a `std::shared_ptr` to the layer.
     */
    class Builder;

public:
    /**
     * @brief Construct a particle layer from a simulation system and an optional color.
     *
     * @param system Host-side shared pointer to the simulation system providing particle data.
     * @param color RGBA color used when rendering particles.
     */
    ATLAS_HOST ParticleLayer(const atlas::SystemHostPtr<T>& system,
                             const Vector4<T>& color = Vector4<T>(T(0.15), T(0.45), T(0.95), T(0.65)));

    /**
     * @brief Create a fluent builder for @ref ParticleLayer.
     *
     * @return A default-initialized builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Destructor.
     */
    ~ParticleLayer() override = default;

    /**
     * @brief Initialize GPU resources and shader state for particle rendering.
     *
     * @details
     * Typical initialization tasks include:
     * - validating the bound system,
     * - creating the VAO and VBO,
     * - constructing or loading the shader program,
     * - resolving MVP and color uniform locations,
     * - preparing backend-specific upload or interop resources.
     *
     * @param window Active GLFW window.
     * @param camera Active Vizkit camera.
     */
    ATLAS_HOST void
    init(GLFWwindow* window, Camera& camera) override;

    /**
     * @brief Update the particle layer for the current frame.
     *
     * @details
     * Typical per-frame work includes:
     * - reading the current active particle positions from the bound system,
     * - resizing visualization buffers if particle capacity has changed,
     * - uploading or mapping particle positions into the VBO,
     * - preparing the MVP matrix using the active camera,
     * - updating the draw count to match the current active particle count.
     *
     * @param window Active GLFW window.
     * @param camera Active Vizkit camera.
     * @param dt Frame timestep.
     */
    ATLAS_HOST void
    update(GLFWwindow* window, Camera& camera, T dt) override;

    /**
     * @brief Release rendering resources owned by the particle layer.
     *
     * @details
     * Typical shutdown work includes:
     * - unregistering CUDA/OpenGL interop resources when applicable,
     * - deleting the VAO and VBO,
     * - releasing the shader program,
     * - clearing transient host-side staging data when appropriate.
     */
    ATLAS_HOST void
    shutdown() override;

private:
    /**
     * @brief Validate that the bound system is usable for particle rendering.
     *
     * @details
     * Typical validation checks may include:
     * - the system pointer is non-null,
     * - the system contains a valid fluid object,
     * - particle-position buffers are available,
     * - the runtime state is suitable for rendering.
     *
     * The exact validation policy is implementation-defined in
     * `particle_layer.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate_system() const;

    /**
     * @brief Return the OpenGL vertex component type corresponding to `T`.
     *
     * @details
     * This helper maps the scalar coordinate type to the OpenGL enum used when
     * specifying the vertex attribute layout for particle positions.
     *
     * Typical mappings include:
     * - `float`  -> `GL_FLOAT`
     * - `double` -> `GL_DOUBLE`
     *
     * @return OpenGL enum describing the vertex component type.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE GLenum
    vertex_component_type() const noexcept;

private:
    /**
     * @brief Bound simulation system providing particle data.
     *
     * @details
     * This system is the authoritative runtime data source for:
     * - active particle positions,
     * - active particle count,
     * - buffer capacity information.
     */
    atlas::SystemHostPtr<T> _system {};

    /**
     * @brief OpenGL vertex array object handle.
     *
     * @details
     * Encapsulates the vertex attribute binding state used for particle rendering.
     */
    GLuint _vao = 0;

    /**
     * @brief OpenGL vertex buffer object handle.
     *
     * @details
     * Stores uploaded particle positions for rendering.
     */
    GLuint _vbo = 0;

    /**
     * @brief Cached uniform location for the MVP matrix.
     */
    GLint _u_mvp = -1;

    /**
     * @brief Cached uniform location for the render color.
     */
    GLint _u_color = -1;

    /**
     * @brief Number of particles currently scheduled for drawing.
     *
     * @details
     * This usually matches the active particle count exposed by the system at
     * the last update.
     */
    int _draw_count = 0;

    /**
     * @brief Currently allocated visualization capacity.
     *
     * @details
     * Tracks how many particle positions the current rendering buffer can hold
     * without requiring reallocation.
     */
    std::size_t _capacity = 0;

    /**
     * @brief RGBA color used when rendering particles.
     *
     * @details
     * The first three components define the displayed color and the fourth
     * component defines transparency.
     */
    Vector4<T> _color { T(0.15), T(0.45), T(0.95), T(0.65) };

    /**
     * @brief Owned shader program used to render the particle cloud.
     */
    std::unique_ptr<ShaderProgram> _program;

#if defined(ATLAS_TASKING_CUDA)
    /**
     * @brief CUDA/OpenGL interop handle for the particle VBO.
     *
     * @details
     * Present only in CUDA-enabled builds. This resource allows the vertex
     * buffer to participate in CUDA graphics interop workflows.
     */
    cudaGraphicsResource* _cuda_vbo_resource = nullptr;
#else
    /**
     * @brief Host-side staging buffer for particle positions.
     *
     * @details
     * Present only in non-CUDA builds. The current particle positions are copied
     * here before being uploaded into the OpenGL vertex buffer.
     */
    std::vector<Vector3<T>> _host_positions;
#endif
};

/**
 * @brief Fluent builder for @ref ParticleLayer.
 *
 * @details
 * The builder provides a controlled construction path for the particle layer by
 * staging:
 * - the simulation system to visualize,
 * - the RGBA render color.
 *
 * ## Typical usage
 * @code
 * auto layer = atlas::vizkit::ParticleLayer<float>::builder()
 *     .with_system(system)
 *     .with_color({0.2f, 0.6f, 1.0f, 0.8f})
 *     .make_shared();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_shared. Typical checks may include:
 * - the system pointer is non-null,
 * - the system is valid for visualization,
 * - the color contents are acceptable for rendering.
 *
 * The exact validation rules are implementation-defined in `particle_layer.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar used for particle coordinates and color.
 */
template <typename T>
class ParticleLayer<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the builder with:
     * - no target system,
     * - a default semi-transparent blue color.
     */
    Builder() = default;

    /**
     * @brief Set the simulation system to visualize.
     *
     * @param system Host-side shared pointer to the source system.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_system(const atlas::SystemHostPtr<T>& system) noexcept;

    /**
     * @brief Set the particle render color.
     *
     * @param color RGBA color to stage for rendering.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_color(const Vector4<T>& color) noexcept;

    /**
     * @brief Build a configured @ref ParticleLayer by value after validation.
     *
     * @return Constructed particle layer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE ParticleLayer<T>
    build() const;

    /**
     * @brief Build a configured @ref ParticleLayer in shared ownership after validation.
     *
     * @return `std::shared_ptr<ParticleLayer<T>>` owning the constructed layer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE std::shared_ptr<ParticleLayer<T>>
    make_shared() const;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the staged system and render color.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending simulation system dependency.
     */
    atlas::SystemHostPtr<T> _system {};

    /**
     * @brief Pending particle render color.
     */
    Vector4<T> _color { T(0.15), T(0.45), T(0.95), T(0.65) };
};

} // namespace atlas::vizkit

#include <vizkit/layer/particle/particle_layer.hpp>

#endif