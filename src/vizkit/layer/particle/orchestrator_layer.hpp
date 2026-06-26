#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <vizkit/shader/glsl.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace atlas::vizkit {

template <typename T>
OrchestratorLayer<T>::OrchestratorLayer(const atlas::OrchestratorHostPtr<T>& orchestrator,
                                        const atlas::Vector4<float>& default_color,
                                        std::vector<atlas::Vector4<float>> solver_colors,
                                        const T point_size)
    // Store the orchestrator that supplies the simulation-side objects used by
    // this visualization layer, such as:
    //   - the fluid containing particle states,
    //   - the searcher defining cell membership,
    //   - the codec assigning cells to solver partitions,
    //   - the universe defining the cell domain.
    : _orchestrator(orchestrator)

    // Store the fallback color used when no solver-specific coloring is applied.
    , _default_color(default_color)

    // Store the palette used to visualize solver allocations.
    //
    // Each solver index is mapped to one entry in this palette, typically by
    // modulo indexing when the number of solver partitions exceeds the palette size.
    , _solver_colors(std::move(solver_colors))

    // Store the OpenGL point size used for particle rendering.
    , _point_size(point_size) {
    // Restrict this layer to floating-point particle coordinate types.
    //
    // Rendering positions from integer or other non-floating types would not be
    // meaningful for this visualization path.
    static_assert(std::is_floating_point_v<T>, "OrchestratorLayer requires a floating-point T");
}

template <typename T>
typename OrchestratorLayer<T>::Builder
OrchestratorLayer<T>::builder() noexcept {
    // Return a fresh builder with default-initialized configuration fields.
    return Builder {};
}

template <typename T>
void
OrchestratorLayer<T>::init(GLFWwindow* window, Camera& camera) {
    // This layer does not require the window or camera during initialization.
    //
    // They are still accepted to satisfy the common layer interface.
    (void)window;
    (void)camera;

    // Validate that the orchestrator and its fluid dependency exist before any
    // rendering resources are created.
    validate_orchestrator();

    // Use the fluid buffer capacity as the maximum number of points this layer
    // can upload and render without reallocating GPU resources.
    _capacity = _orchestrator->fluid()->buffer_size();

    // No particles are drawn until the first successful update fills the buffers.
    _draw_count = 0;

    // Install a default solver-color palette when the caller did not provide one.
    //
    // These colors are chosen to be visually distinct when showing solver
    // partition ownership across particles.
    if (_solver_colors.empty()) {
        _solver_colors = {
            atlas::Vector4<float>(0.13f, 0.62f, 0.95f, 0.92f),
            atlas::Vector4<float>(0.23f, 0.83f, 0.62f, 0.92f),
            atlas::Vector4<float>(0.96f, 0.71f, 0.22f, 0.92f),
            atlas::Vector4<float>(0.93f, 0.33f, 0.22f, 0.92f),
        };
    }

    // Create the shader program used for point rendering.
    //
    // The shaders are expected to:
    //   - transform input positions with the MVP matrix,
    //   - consume a per-vertex RGBA color,
    //   - render each particle as an OpenGL point.
    _program = std::make_unique<ShaderProgram>(k_point_color_vs, k_point_color_fs);

    // Create the OpenGL vertex array and the two vertex buffers:
    //   - one for particle positions,
    //   - one for particle colors.
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_position_vbo);
    glGenBuffers(1, &_color_vbo);

    glBindVertexArray(_vao);

    // Allocate the position buffer with enough storage for the full particle capacity.
    //
    // The buffer is marked dynamic because its contents are updated every frame.
    glBindBuffer(GL_ARRAY_BUFFER, _position_vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(_capacity * sizeof(Vector3<T>)),
        nullptr,
        GL_DYNAMIC_DRAW);

    // Describe vertex attribute 0 as a packed 3-component position.
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        vertex_component_type(),
        GL_FALSE,
        sizeof(Vector3<T>),
        reinterpret_cast<void*>(0));

    // Allocate the color buffer with enough storage for one RGBA color per particle.
    glBindBuffer(GL_ARRAY_BUFFER, _color_vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(_capacity * sizeof(atlas::Vector4<float>)),
        nullptr,
        GL_DYNAMIC_DRAW);

    // Describe vertex attribute 1 as a packed 4-component float color.
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1,
        4,
        GL_FLOAT,
        GL_FALSE,
        sizeof(atlas::Vector4<float>),
        reinterpret_cast<void*>(0));

    // Unbind GL objects so initialization leaves a clean OpenGL state.
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Cache uniform locations used every frame during rendering.
    _u_mvp        = _program->uniform_loc("MVP");
    _u_point_size = _program->uniform_loc("uPointSize");

    // Allocate a device-side color buffer initialized to the default color.
    //
    // This buffer is repainted each frame depending on solver ownership.
    _device_colors.resize(_capacity, _default_color);

    // Upload the solver palette into device memory for CUDA-side coloring.
    _device_solver_colors = DeviceBuffer<atlas::Vector4<float>>(_solver_colors.begin(), _solver_colors.end());

#if defined(ATLAS_TASKING_CUDA)
    // Register the OpenGL position VBO as a CUDA graphics resource so positions
    // can be copied directly from device memory into the GL buffer without a
    // host round-trip.
    const cudaError_t position_register_error = cudaGraphicsGLRegisterBuffer(
        &_cuda_position_vbo_resource,
        _position_vbo,
        cudaGraphicsRegisterFlagsWriteDiscard);

    // Tear down partially initialized resources if CUDA-GL registration fails.
    if (position_register_error != cudaSuccess) {
        shutdown();
        throw std::runtime_error(
            std::string("OrchestratorLayer: failed to register CUDA-GL position buffer: ")
            + cudaGetErrorString(position_register_error));
    }

    // Register the OpenGL color VBO as a CUDA graphics resource for the same reason.
    const cudaError_t color_register_error = cudaGraphicsGLRegisterBuffer(
        &_cuda_color_vbo_resource,
        _color_vbo,
        cudaGraphicsRegisterFlagsWriteDiscard);

    if (color_register_error != cudaSuccess) {
        shutdown();
        throw std::runtime_error(
            std::string("OrchestratorLayer: failed to register CUDA-GL color buffer: ")
            + cudaGetErrorString(color_register_error));
    }
#else
    // In the non-CUDA path, allocate host-side staging buffers used to copy
    // particle positions and colors from device memory to the CPU before
    // uploading them into OpenGL with glBufferSubData.
    _host_positions.resize(_capacity);
    _host_colors.resize(_capacity, _default_color);
#endif
}

template <typename T>
void
OrchestratorLayer<T>::update(GLFWwindow* window, Camera& camera, T dt) {
    // This layer does not use the simulation time step directly during rendering.
    (void)dt;

    // Do nothing if the required rendering objects or simulation dependencies
    // have not been initialized.
    if (!_program || !_vao || !_position_vbo || !_color_vbo || !_orchestrator || !_orchestrator->fluid()) {
        return;
    }

    const auto& fluid = _orchestrator->fluid();

    // Retrieve the particle position state to visualize.
    const auto* position_state = fluid->template state<atlas::FluidPositionState<T>>();

    // Nothing can be rendered when the position state is missing or when no
    // active particles exist.
    if (position_state == nullptr || fluid->particle_count() == 0) {
        return;
    }

    // Synchronize all simulation-side views that influence the coloring and
    // spatial interpretation of particles.
    //
    // This updates:
    //   - the searcher, so cell membership is current,
    //   - the measurer, if present,
    //   - the codec, so allocated_solver() reflects the latest partitioning.
    sync_codec_view();

    // Clamp the active particle count to the preallocated rendering capacity.
    const std::size_t active_count = std::min<std::size_t>(fluid->particle_count(), _capacity);
    if (active_count == 0) {
        return;
    }

    // Access the particle positions in device memory.
    const auto* particle_pos = atlas::raw_pointer_cast(position_state->data().data());

    // Cache orchestrator-owned helpers used for optional solver-based coloring.
    const auto& searcher = _orchestrator->searcher();
    const auto& codec    = _orchestrator->codec();
    const auto& universe = _orchestrator->universe();

#if defined(ATLAS_TASKING_CUDA)
    // Reset all particle colors to the default value before optionally painting
    // solver-specific colors over the active subset.
    atlas::parallel_fill<ExecutionPolicy::device>(
        _device_colors.begin(),
        _device_colors.begin() + static_cast<std::ptrdiff_t>(active_count),
        _default_color);

    // If all supporting data exists, recolor particles according to the solver
    // allocation of the cell they currently belong to.
    if (searcher && codec && universe && !_device_solver_colors.empty()) {
        const int cell_count = universe->number_of_cells();
        if (cell_count > 0) {
            auto* device_colors_ptr          = atlas::raw_pointer_cast(_device_colors.data());
            const auto* indices_ptr          = searcher->indices();
            const auto* cell_start_ptr       = searcher->cell_start();
            const auto* cell_end_ptr         = searcher->cell_end();
            const auto* allocated_solver_ptr = atlas::raw_pointer_cast(codec->allocated_solver().data());
            const auto* palette_ptr          = atlas::raw_pointer_cast(_device_solver_colors.data());
            const int palette_size           = static_cast<int>(_device_solver_colors.size());
            const int particle_count_int     = static_cast<int>(active_count);

            // Process one cell per device work item and paint all particles in
            // that cell with the color assigned to the cell's solver index.
            atlas::parallel_for<ExecutionPolicy::device>(
                0,
                cell_count,
                [=] ATLAS_DEVICE(const int cell) {
                    const int start = cell_start_ptr[cell];
                    const int end   = cell_end_ptr[cell];

                    // Skip invalid or empty cell ranges.
                    if (start < 0 || end <= start) {
                        return;
                    }

                    // Map the solver index into the palette range using modular arithmetic.
                    //
                    // The signed-safe normalization allows negative solver indices
                    // to still map deterministically into the palette.
                    const int solver_index            = allocated_solver_ptr[cell];
                    const int palette_index           = palette_size > 0
                                  ? ((solver_index % palette_size) + palette_size) % palette_size
                                  : 0;
                    const atlas::Vector4<float> color = palette_ptr[palette_index];

                    // Paint every valid particle in the cell with the chosen solver color.
                    for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                        const int particle_index = indices_ptr[sorted_index];
                        if (particle_index >= 0 && particle_index < particle_count_int) {
                            device_colors_ptr[particle_index] = color;
                        }
                    }
                });
        }
    }

    // The CUDA path requires successful registration of both GL resources.
    if (_cuda_position_vbo_resource == nullptr || _cuda_color_vbo_resource == nullptr) {
        return;
    }

    // Map both OpenGL buffers into CUDA address space for direct device writes.
    cudaGraphicsMapResources(1, &_cuda_position_vbo_resource, 0);
    cudaGraphicsMapResources(1, &_cuda_color_vbo_resource, 0);

    void* mapped_position_buffer      = nullptr;
    void* mapped_color_buffer         = nullptr;
    std::size_t mapped_position_bytes = 0;
    std::size_t mapped_color_bytes    = 0;

    // Query the mapped CUDA pointers and their accessible byte sizes.
    cudaGraphicsResourceGetMappedPointer(
        &mapped_position_buffer,
        &mapped_position_bytes,
        _cuda_position_vbo_resource);
    cudaGraphicsResourceGetMappedPointer(
        &mapped_color_buffer,
        &mapped_color_bytes,
        _cuda_color_vbo_resource);

    const std::size_t required_position_bytes = active_count * sizeof(Vector3<T>);
    const std::size_t required_color_bytes    = active_count * sizeof(atlas::Vector4<float>);

    // Copy particle positions directly from device simulation memory into the
    // mapped OpenGL position buffer.
    if (mapped_position_buffer != nullptr && mapped_position_bytes >= required_position_bytes) {
        cudaMemcpy(
            mapped_position_buffer,
            particle_pos,
            required_position_bytes,
            cudaMemcpyDeviceToDevice);
    }

    // Copy the computed device-side particle colors into the mapped OpenGL color buffer.
    if (mapped_color_buffer != nullptr && mapped_color_bytes >= required_color_bytes) {
        cudaMemcpy(
            mapped_color_buffer,
            atlas::raw_pointer_cast(_device_colors.data()),
            required_color_bytes,
            cudaMemcpyDeviceToDevice);
    }

    // Unmap the OpenGL resources so the graphics pipeline can consume them.
    cudaGraphicsUnmapResources(1, &_cuda_color_vbo_resource, 0);
    cudaGraphicsUnmapResources(1, &_cuda_position_vbo_resource, 0);
#else
    // In the host fallback path, first copy particle positions from device to host memory.
    atlas::copy_device_to_host(particle_pos, _host_positions.data(), active_count);

    // Reset all host-side colors to the default before applying solver coloring.
    std::fill(_host_colors.begin(), _host_colors.begin() + static_cast<std::ptrdiff_t>(active_count), _default_color);

    // If cell ownership data is available, recolor host-side particles based on
    // the solver assignment of their current cell.
    if (searcher && codec && universe) {
        const int cell_count = universe->number_of_cells();
        if (cell_count > 0) {
            // Allocate host-side staging buffers for the searcher and codec data.
            _host_indices.resize(active_count);
            _host_cell_start.resize(static_cast<std::size_t>(cell_count));
            _host_cell_end.resize(static_cast<std::size_t>(cell_count));
            _host_allocated_solver.resize(static_cast<std::size_t>(cell_count));

            // Copy cell topology and allocation data from device to host.
            atlas::copy_device_to_host(searcher->indices(), _host_indices.data(), active_count);
            atlas::copy_device_to_host(searcher->cell_start(), _host_cell_start.data(), static_cast<std::size_t>(cell_count));
            atlas::copy_device_to_host(searcher->cell_end(), _host_cell_end.data(), static_cast<std::size_t>(cell_count));
            atlas::copy_device_to_host(
                atlas::raw_pointer_cast(codec->allocated_solver().data()),
                _host_allocated_solver.data(),
                static_cast<std::size_t>(cell_count));

            // Recolor particles cell by cell on the host.
            for (int cell = 0; cell < cell_count; ++cell) {
                const int start = _host_cell_start[static_cast<std::size_t>(cell)];
                const int end   = _host_cell_end[static_cast<std::size_t>(cell)];

                if (start < 0 || end <= start) {
                    continue;
                }

                // Choose a palette color for the cell's solver index.
                //
                // Negative solver indices are clamped to zero in this host path.
                const int solver_index            = _host_allocated_solver[static_cast<std::size_t>(cell)];
                const std::size_t palette_index   = _solver_colors.empty()
                      ? 0u
                      : static_cast<std::size_t>(std::max(solver_index, 0)) % _solver_colors.size();
                const atlas::Vector4<float> color = _solver_colors.empty() ? _default_color : _solver_colors[palette_index];

                for (int sorted_index = start; sorted_index < end; ++sorted_index) {
                    if (sorted_index < 0 || static_cast<std::size_t>(sorted_index) >= _host_indices.size()) {
                        continue;
                    }

                    const int particle_index = _host_indices[static_cast<std::size_t>(sorted_index)];
                    if (particle_index < 0 || static_cast<std::size_t>(particle_index) >= active_count) {
                        continue;
                    }

                    _host_colors[static_cast<std::size_t>(particle_index)] = color;
                }
            }
        }
    }

    // Upload the host-side positions into the OpenGL position VBO.
    glBindBuffer(GL_ARRAY_BUFFER, _position_vbo);
    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        static_cast<GLsizeiptr>(active_count * sizeof(Vector3<T>)),
        _host_positions.data());

    // Upload the host-side colors into the OpenGL color VBO.
    glBindBuffer(GL_ARRAY_BUFFER, _color_vbo);
    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        static_cast<GLsizeiptr>(active_count * sizeof(atlas::Vector4<float>)),
        _host_colors.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif

    // Record how many points will be submitted in the draw call.
    _draw_count = static_cast<int>(active_count);

    // Query the framebuffer size so the camera can construct a correct MVP matrix.
    int width  = 1;
    int height = 1;
    glfwGetFramebufferSize(window, &width, &height);

    float mvp[16];
    camera.build_mvp(width, height, mvp);

    // Configure point rendering state.
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Bind the shader and upload per-frame uniforms.
    _program->use();
    glUniformMatrix4fv(_u_mvp, 1, GL_FALSE, mvp);
    if (_u_point_size >= 0) {
        glUniform1f(_u_point_size, static_cast<float>(_point_size));
    }

    // Draw one GL point per active particle.
    glBindVertexArray(_vao);
    glDrawArrays(GL_POINTS, 0, _draw_count);
    glBindVertexArray(0);
}

template <typename T>
void
OrchestratorLayer<T>::shutdown() {
#if defined(ATLAS_TASKING_CUDA)
    // Unregister CUDA graphics resources before deleting the underlying GL buffers.
    if (_cuda_color_vbo_resource != nullptr) {
        cudaGraphicsUnregisterResource(_cuda_color_vbo_resource);
        _cuda_color_vbo_resource = nullptr;
    }

    if (_cuda_position_vbo_resource != nullptr) {
        cudaGraphicsUnregisterResource(_cuda_position_vbo_resource);
        _cuda_position_vbo_resource = nullptr;
    }
#endif

    // Delete the OpenGL color buffer if it exists.
    if (_color_vbo) {
        glDeleteBuffers(1, &_color_vbo);
        _color_vbo = 0;
    }

    // Delete the OpenGL position buffer if it exists.
    if (_position_vbo) {
        glDeleteBuffers(1, &_position_vbo);
        _position_vbo = 0;
    }

    // Delete the vertex array object if it exists.
    if (_vao) {
        glDeleteVertexArrays(1, &_vao);
        _vao = 0;
    }

    // Reset cached GL-related state.
    _u_mvp        = -1;
    _u_point_size = -1;
    _draw_count   = 0;
    _capacity     = 0;

#if !defined(ATLAS_TASKING_CUDA)
    // Release host-side staging buffers used only in the non-CUDA fallback path.
    _host_positions.clear();
    _host_colors.clear();
    _host_indices.clear();
    _host_cell_start.clear();
    _host_cell_end.clear();
    _host_allocated_solver.clear();
#endif

    // Release device-side color buffers.
    _device_colors.clear();
    _device_solver_colors.clear();

    // Destroy the shader program last.
    _program.reset();
}

template <typename T>
void
OrchestratorLayer<T>::validate_orchestrator() const {
    // The visualization layer requires a valid orchestrator object.
    if (_orchestrator == nullptr) {
        throw std::runtime_error("OrchestratorLayer: orchestrator must not be null.");
    }

    // The orchestrator must provide a valid fluid because particle rendering
    // depends on fluid-owned position buffers and capacity information.
    if (_orchestrator->fluid() == nullptr) {
        throw std::runtime_error("OrchestratorLayer: orchestrator fluid must not be null.");
    }
}

template <typename T>
GLenum
OrchestratorLayer<T>::vertex_component_type() const noexcept {
    // Match the OpenGL vertex attribute scalar type to the template parameter T.
    if constexpr (std::is_same_v<T, double>) {
        return GL_DOUBLE;
    }

    return GL_FLOAT;
}

template <typename T>
void
OrchestratorLayer<T>::sync_codec_view() {
    const auto& searcher = _orchestrator->searcher();
    if (searcher) {
        // Rebuild the particle-to-cell organization before reading any cell-local data.
        searcher->build();
    }

    const auto& measurer = _orchestrator->measurer();
    if (measurer) {
        // Refresh measurement-dependent orchestration state if a measurer is available.
        measurer->measure();
    }

    const auto& codec = _orchestrator->codec();
    if (codec) {
        // Refresh the codec so allocated_solver() reflects the latest partition assignment.
        codec->update();
    }
}

template <typename T>
typename OrchestratorLayer<T>::Builder&
OrchestratorLayer<T>::Builder::with_orchestrator(const atlas::OrchestratorHostPtr<T>& orchestrator) noexcept {
    // Store the orchestrator dependency in the builder.
    _orchestrator = orchestrator;
    return *this;
}

template <typename T>
typename OrchestratorLayer<T>::Builder&
OrchestratorLayer<T>::Builder::with_default_color(const atlas::Vector4<float>& color) noexcept {
    // Store the fallback particle color in the builder.
    _default_color = color;
    return *this;
}

template <typename T>
typename OrchestratorLayer<T>::Builder&
OrchestratorLayer<T>::Builder::with_solver_colors(const std::vector<atlas::Vector4<float>>& solver_colors) {
    // Store the solver-color palette in the builder.
    _solver_colors = solver_colors;
    return *this;
}

template <typename T>
typename OrchestratorLayer<T>::Builder&
OrchestratorLayer<T>::Builder::with_point_size(const T point_size) noexcept {
    // Store the desired point size in the builder.
    _point_size = point_size;
    return *this;
}

template <typename T>
OrchestratorLayer<T>
OrchestratorLayer<T>::Builder::build() const {
    // Validate the builder state before constructing a value instance.
    validate();
    return OrchestratorLayer<T>(_orchestrator, _default_color, _solver_colors, _point_size);
}

template <typename T>
std::shared_ptr<OrchestratorLayer<T>>
OrchestratorLayer<T>::Builder::make_shared() const {
    // Validate the builder state before constructing a shared instance.
    validate();
    return std::make_shared<OrchestratorLayer<T>>(_orchestrator, _default_color, _solver_colors, _point_size);
}

template <typename T>
void
OrchestratorLayer<T>::Builder::validate() const {
    // A valid orchestrator is required.
    if (_orchestrator == nullptr) {
        throw std::runtime_error("OrchestratorLayer::Builder: orchestrator must not be null.");
    }

    // The orchestrator must expose a valid fluid dependency.
    if (_orchestrator->fluid() == nullptr) {
        throw std::runtime_error("OrchestratorLayer::Builder: orchestrator fluid must not be null.");
    }

    // Point size must be finite and strictly positive for valid rendering.
    if (!std::isfinite(_point_size) || _point_size <= T(0)) {
        throw std::runtime_error("OrchestratorLayer::Builder: point_size must be finite and positive.");
    }
}

} // namespace atlas::vizkit

#endif