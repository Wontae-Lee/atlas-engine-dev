#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file orchestrator_layer.h
 * @brief Declares a visualization layer that colors particles according to orchestrator solver allocation.
 *
 * This layer renders particles managed by an `atlas::Orchestrator<T>` instance
 * and colors each particle based on which solver is currently responsible for
 * its cell (as determined by the orchestrator's codec).
 *
 * The primary purpose of this layer is to:
 * - visualize solver-region partitioning in hybrid simulations
 * - highlight transitions between SPH and DSMC regimes
 * - debug codec-based solver assignment
 *
 * The layer supports both:
 * - GPU interop path (CUDA + OpenGL VBO interop)
 * - CPU fallback path (host-side staging buffers)
 */

#include <atlas/core/macros.h>
#include <atlas/buffer/device_buffer.h>
#include <atlas/math/vector/vector4.h>
#include <atlas/orchestrator/orchestrator.h>

#include <vizkit/layer/layer.h>
#include <vizkit/shader/shader_program.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

/**
 * @brief Visualization layer that colors particles based on orchestrator solver assignment.
 *
 * This layer pulls particle data from an `atlas::Orchestrator<T>` instance and
 * assigns a color per particle based on:
 * - the solver index assigned to the particle's cell
 * - an optional user-provided solver-color palette
 *
 * Rendering pipeline overview:
 * 1. Query particle positions and cell assignments from the orchestrator
 * 2. Map each particle to its solver index via codec allocation
 * 3. Assign color based on solver index or fallback color
 * 4. Upload positions and colors to GPU buffers
 * 5. Render particles using point-based rendering
 *
 * The layer internally maintains:
 * - OpenGL vertex array object (VAO)
 * - position vertex buffer (VBO)
 * - color vertex buffer (VBO)
 * - optional CUDA graphics resources for zero-copy updates
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class OrchestratorLayer final : public Layer<T> {
public:
    class Builder;

public:
    /**
     * @brief Construct the orchestrator visualization layer.
     *
     * @param orchestrator Source orchestrator providing particle data and solver assignments.
     * @param default_color Fallback color used when no solver-specific color is available.
     * @param solver_colors Optional list of colors mapped by solver index.
     * @param point_size Size of rendered particles.
     */
    ATLAS_HOST OrchestratorLayer(
        const atlas::OrchestratorHostPtr<T>& orchestrator,
        const atlas::Vector4<float>& default_color = atlas::Vector4<float>(0.15f, 0.45f, 0.95f, 0.65f),
        std::vector<atlas::Vector4<float>> solver_colors = {},
        T point_size = T(3.0));

    /**
     * @brief Create a builder for `OrchestratorLayer`.
     *
     * @return Default-initialized builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ~OrchestratorLayer() override = default;

    /**
     * @brief Initialize rendering resources.
     *
     * This function:
     * - validates the orchestrator
     * - creates OpenGL VAO and VBOs
     * - compiles and binds shader program
     * - prepares GPU-side buffers
     *
     * @param window GLFW window context.
     * @param camera Active camera.
     */
    ATLAS_HOST void
    init(GLFWwindow* window, Camera& camera) override;

    /**
     * @brief Update particle buffers and render frame.
     *
     * This function:
     * - synchronizes particle data from the orchestrator
     * - updates position and color buffers
     * - issues draw calls for all particles
     *
     * @param window GLFW window context.
     * @param camera Active camera.
     * @param dt Frame time step (not used for simulation here, only visualization timing).
     */
    ATLAS_HOST void
    update(GLFWwindow* window, Camera& camera, T dt) override;

    /**
     * @brief Release rendering resources.
     *
     * This function cleans up:
     * - OpenGL buffers
     * - CUDA interop resources (if enabled)
     * - shader program
     */
    ATLAS_HOST void
    shutdown() override;

private:
    /**
     * @brief Validate that the orchestrator is correctly initialized.
     *
     * Ensures that:
     * - the orchestrator pointer is not null
     * - required internal data (fluid, searcher, codec) is accessible
     *
     * @throws std::runtime_error if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate_orchestrator() const;

    /**
     * @brief Return the OpenGL component type used for vertex buffers.
     *
     * This depends on the scalar type `T` and ensures correct interpretation
     * by the GPU.
     *
     * @return OpenGL enum representing the vertex component type.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE GLenum
    vertex_component_type() const noexcept;

    /**
     * @brief Synchronize particle data and solver assignments from the orchestrator.
     *
     * This function:
     * - fetches particle positions
     * - maps particles to solver indices
     * - assigns colors accordingly
     * - uploads results into device buffers
     *
     * The exact implementation differs between:
     * - CUDA path (direct GPU mapping)
     * - CPU fallback path (host staging buffers)
     */
    ATLAS_HOST void
    sync_codec_view();

private:
    /// Source orchestrator providing simulation data.
    atlas::OrchestratorHostPtr<T> _orchestrator {};

    /// OpenGL vertex array object.
    GLuint _vao = 0;

    /// OpenGL vertex buffer storing particle positions.
    GLuint _position_vbo = 0;

    /// OpenGL vertex buffer storing particle colors.
    GLuint _color_vbo = 0;

    /// Shader uniform for model-view-projection matrix.
    GLint _u_mvp = -1;

    /// Shader uniform controlling point size.
    GLint _u_point_size = -1;

    /// Number of particles to render.
    int _draw_count = 0;

    /// Current allocated capacity of buffers.
    std::size_t _capacity = 0;

    /// Default color used when solver index is not mapped.
    atlas::Vector4<float> _default_color { 0.15f, 0.45f, 0.95f, 0.65f };

    /// Per-solver color palette.
    std::vector<atlas::Vector4<float>> _solver_colors {};

    /// Rendered particle point size.
    T _point_size = T(3.0);

    /// Shader program used for rendering.
    std::unique_ptr<ShaderProgram> _program;

    /// Device buffer storing per-particle colors.
    DeviceBuffer<atlas::Vector4<float>> _device_colors {};

    /// Device buffer storing solver color palette.
    DeviceBuffer<atlas::Vector4<float>> _device_solver_colors {};

#if defined(ATLAS_TASKING_CUDA)
    /// CUDA-OpenGL interop resource for position buffer.
    cudaGraphicsResource* _cuda_position_vbo_resource = nullptr;

    /// CUDA-OpenGL interop resource for color buffer.
    cudaGraphicsResource* _cuda_color_vbo_resource = nullptr;
#else
    /// Host-side particle positions (CPU fallback).
    std::vector<Vector3<T>> _host_positions;

    /// Host-side particle colors (CPU fallback).
    std::vector<atlas::Vector4<float>> _host_colors;

    /// Host-side particle index mapping.
    std::vector<int> _host_indices;

    /// Host-side cell start offsets.
    std::vector<int> _host_cell_start;

    /// Host-side cell end offsets.
    std::vector<int> _host_cell_end;

    /// Host-side solver assignment per cell.
    std::vector<int> _host_allocated_solver;
#endif
};

/**
 * @brief Builder for `OrchestratorLayer`.
 *
 * This builder allows configuration of:
 * - orchestrator source
 * - default particle color
 * - solver-specific color palette
 * - rendering point size
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class OrchestratorLayer<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_orchestrator(const atlas::OrchestratorHostPtr<T>& orchestrator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_default_color(const atlas::Vector4<float>& color) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solver_colors(const std::vector<atlas::Vector4<float>>& solver_colors);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_point_size(T point_size) noexcept;

    /**
     * @brief Build a validated layer instance.
     *
     * @return Constructed orchestrator visualization layer.
     *
     * @throws std::runtime_error if configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE OrchestratorLayer<T>
    build() const;

    /**
     * @brief Build a validated layer and wrap it in a shared pointer.
     *
     * @return Shared pointer to the constructed layer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE std::shared_ptr<OrchestratorLayer<T>>
    make_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    atlas::OrchestratorHostPtr<T> _orchestrator {};
    atlas::Vector4<float> _default_color { 0.15f, 0.45f, 0.95f, 0.65f };
    std::vector<atlas::Vector4<float>> _solver_colors {};
    T _point_size = T(3.0);
};

} // namespace atlas::vizkit

#include <vizkit/layer/particle/orchestrator_layer.hpp>

#endif