#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <atlas/logging/logging.h>
#include <vizkit/shader/glsl.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace atlas::vizkit {

template <typename T>
ParticleLayer<T>::ParticleLayer(const atlas::SystemHostPtr<T>& system,
                                const Vector4<T>& color,
                                const T point_size)
    : _system(system)
    , _color(color)
    , _point_size(point_size) {
    // Construct a particle visualization layer.
    //
    // Parameters:
    // - system : simulation system providing particle data
    // - color  : RGBA color used when rendering particles
    //
    // This constructor only stores references / configuration.
    // GPU resources are allocated later in init().
    static_assert(std::is_floating_point_v<T>, "ParticleLayer requires a floating-point T");
}

template <typename T>
typename ParticleLayer<T>::Builder
ParticleLayer<T>::builder() noexcept {
    // Return a fresh builder for staged ParticleLayer construction.
    return Builder {};
}

template <typename T>
void
ParticleLayer<T>::init(GLFWwindow* window, Camera& camera) {
    // Initialize all rendering resources required to draw particles.
    //
    // High-level workflow:
    // 1. validate the bound simulation system
    // 2. determine maximum drawable particle capacity
    // 3. create shader program
    // 4. allocate VAO/VBO
    // 5. configure vertex attribute layout
    // 6. cache shader uniform locations
    // 7. prepare backend-specific upload path
    //
    // Current implementation does not need window/camera during initialization.
    (void)window;
    (void)camera;

    // Verify that the system and its fluid are valid before allocating resources.
    validate_system();

    // Cache the maximum number of particle positions that can be visualized.
    //
    // This capacity is derived from the fluid buffer size and is used for:
    // - VBO allocation size
    // - host staging buffer size in non-CUDA mode
    _capacity = _system->fluid()->buffer_size();

    // No particles are drawn until the first successful update().
    _draw_count = 0;

    // Create the shader program used for point rendering.
    //
    // Shader pair:
    // - k_point_vs : vertex shader for particles
    // - k_point_fs : fragment shader for particles
    _program = std::make_unique<ShaderProgram>(k_point_vs, k_point_fs);

    // Allocate OpenGL objects:
    // - one VAO for vertex attribute state
    // - one VBO for particle position data
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);

    // Bind the VAO so subsequent attribute configuration is recorded into it.
    glBindVertexArray(_vao);

    // Bind the VBO so storage can be allocated.
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);

    // Allocate GPU buffer storage for the maximum particle capacity.
    //
    // Buffer contents are initially null because actual particle positions
    // are uploaded during update().
    //
    // Usage hint:
    // - GL_DYNAMIC_DRAW because the position data changes every frame
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(_capacity * sizeof(Vector3<T>)),
        nullptr,
        GL_DYNAMIC_DRAW);

    // Enable vertex attribute location 0.
    //
    // Convention used here:
    // - attribute 0 = particle position
    glEnableVertexAttribArray(0);

    // Describe the vertex layout stored in the VBO.
    //
    // Layout:
    // - 3 scalar components per vertex
    // - component type determined by T
    // - tightly packed as Vector3<T>
    glVertexAttribPointer(
        0,
        3,
        vertex_component_type(),
        GL_FALSE,
        sizeof(Vector3<T>),
        reinterpret_cast<void*>(0));

    // Unbind VAO to avoid unintended state modification elsewhere.
    glBindVertexArray(0);

    // Unbind VBO after configuration.
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Cache uniform locations used during rendering.
    //
    // "MVP"   : model-view-projection matrix
    // "uColor": particle RGBA color
    _u_mvp   = _program->uniform_loc("MVP");
    _u_color = _program->uniform_loc("uColor");
    _u_point_size = _program->uniform_loc("uPointSize");

#if defined(ATLAS_TASKING_CUDA)
    // In CUDA mode, register the OpenGL VBO as a CUDA graphics resource.
    //
    // This enables direct device-to-device particle position upload into the VBO
    // without staging through host memory.
    const cudaError_t register_error = cudaGraphicsGLRegisterBuffer(
        &_cuda_vbo_resource,
        _vbo,
        cudaGraphicsRegisterFlagsWriteDiscard);

    // If CUDA-GL interop registration fails:
    // - release already-created resources
    // - surface the failure as an exception
    if (register_error != cudaSuccess) {
        shutdown();
        throw std::runtime_error(
            std::string("ParticleLayer: failed to register CUDA-GL buffer: ")
            + cudaGetErrorString(register_error));
    }
#else
    // In non-CUDA mode, allocate a host staging buffer used to copy particle
    // positions from device memory before uploading them with OpenGL.
    _host_positions.resize(_capacity);
#endif
}

template <typename T>
void
ParticleLayer<T>::update(GLFWwindow* window, Camera& camera, T dt) {
    // Refresh particle position data and render all currently active particles.
    //
    // High-level workflow:
    // 1. verify rendering resources and system availability
    // 2. read active particle count from the simulation
    // 3. clamp active count to visual capacity
    // 4. upload particle positions to the VBO
    // 5. build MVP matrix from current camera/framebuffer
    // 6. configure point rendering state
    // 7. bind shader uniforms and issue GL_POINTS draw call
    //
    // Current rendering path does not use dt directly.
    (void)dt;

    // Rendering cannot proceed if:
    // - program is missing
    // - VAO is missing
    // - VBO is missing
    // - system is missing
    if (!_program || !_vao || !_vbo || !_system) return;

    const auto& fluid = _system->fluid();

    if (!fluid) return;

    const auto* position_state = fluid->template state<atlas::fluid::FluidPositionState<T>>();

    if (position_state == nullptr || fluid->particle_count() == 0) return;

    const auto* particle_pos = atlas::raw_pointer_cast(position_state->data().data());
    const auto particle_count = fluid->particle_count();

    // Clamp the active particle count to the preallocated visualization capacity.
    //
    // This prevents overrunning the VBO when the runtime particle count exceeds
    // the buffer size known during initialization.
    const std::size_t active_count = std::min<std::size_t>(
        static_cast<std::size_t>(particle_count),
        _capacity);

    // If nothing remains after clamping, there is nothing to draw.
    if (active_count == 0) return;

#if defined(ATLAS_TASKING_CUDA)
    // In CUDA mode, particle positions are copied directly from device memory
    // into the OpenGL VBO via CUDA-GL interop.

    // Abort if the VBO is not registered as a CUDA graphics resource.
    if (_cuda_vbo_resource == nullptr) return;

    // Map the OpenGL buffer so CUDA can write into it.
    cudaGraphicsMapResources(1, &_cuda_vbo_resource, 0);

    // Retrieve the raw mapped pointer and available byte size.
    void* mapped_buffer      = nullptr;
    std::size_t mapped_bytes = 0;
    cudaGraphicsResourceGetMappedPointer(&mapped_buffer, &mapped_bytes, _cuda_vbo_resource);

    // Compute the number of bytes required for the current active particle prefix.
    const std::size_t required_bytes = active_count * sizeof(Vector3<T>);

    // Perform device-to-device copy only if the mapping succeeded and the mapped
    // buffer is large enough.
    if (mapped_buffer != nullptr && mapped_bytes >= required_bytes) {
        cudaMemcpy(mapped_buffer, particle_pos, required_bytes, cudaMemcpyDeviceToDevice);
    }

    // Unmap the graphics resource so OpenGL can use the updated VBO contents.
    cudaGraphicsUnmapResources(1, &_cuda_vbo_resource, 0);
#else
    // In non-CUDA mode, copy active particle positions from device memory to a
    // host staging buffer first.
    atlas::copy_device_to_host(particle_pos, _host_positions.data(), active_count);

    // Bind the VBO so its contents can be updated from host memory.
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);

    // Upload only the active prefix of particle positions.
    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        static_cast<GLsizeiptr>(active_count * sizeof(Vector3<T>)),
        _host_positions.data());

    // Unbind the VBO after update.
    glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif

    // Cache the number of particles that will be drawn this frame.
    _draw_count = static_cast<int>(active_count);

    // Query framebuffer size so the camera can build an aspect-correct MVP matrix.
    int width  = 1;
    int height = 1;
    glfwGetFramebufferSize(window, &width, &height);

    // Build the current model-view-projection matrix.
    float mvp[16];
    camera.build_mvp(width, height, mvp);

    // Enable programmable point sizing so the shader can control point size if needed.
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Enable alpha blending for particle rendering.
    //
    // Blend mode:
    // - standard source-alpha over destination
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Activate the particle shader program.
    _program->use();

    // Upload the MVP transform.
    glUniformMatrix4fv(_u_mvp, 1, GL_FALSE, mvp);

    // Upload particle color if the shader exposes the uniform.
    if (_u_color >= 0) {
        glUniform4f(
            _u_color,
            static_cast<float>(_color.x),
            static_cast<float>(_color.y),
            static_cast<float>(_color.z),
            static_cast<float>(_color.w));
    }

    if (_u_point_size >= 0) {
        glUniform1f(_u_point_size, static_cast<float>(_point_size));
    }

    // Bind the VAO containing particle vertex layout state.
    glBindVertexArray(_vao);

    // Draw all active particles as points.
    glDrawArrays(GL_POINTS, 0, _draw_count);

    // Unbind the VAO after drawing.
    glBindVertexArray(0);
}

template <typename T>
void
ParticleLayer<T>::shutdown() {
    // Release all backend resources and reset layer state.
    //
    // After shutdown():
    // - particle rendering resources are gone
    // - draw count and capacity are reset
    // - a future init() call is required before rendering again
#if defined(ATLAS_TASKING_CUDA)
    // In CUDA mode, unregister CUDA-GL interop resource first.
    if (_cuda_vbo_resource != nullptr) {
        cudaGraphicsUnregisterResource(_cuda_vbo_resource);
        _cuda_vbo_resource = nullptr;
    }
#else
    // In non-CUDA mode, clear the host staging buffer.
    _host_positions.clear();
#endif

    // Delete the particle VBO if it exists.
    if (_vbo) {
        glDeleteBuffers(1, &_vbo);
        _vbo = 0;
    }

    // Delete the particle VAO if it exists.
    if (_vao) {
        glDeleteVertexArrays(1, &_vao);
        _vao = 0;
    }

    // Reset cached uniform locations.
    _u_mvp        = -1;
    _u_color      = -1;
    _u_point_size = -1;

    // Reset draw state.
    _draw_count = 0;
    _capacity   = 0;

    // Destroy the shader program.
    _program.reset();
}

template <typename T>
void
ParticleLayer<T>::validate_system() const {
    // Validate that the bound system is usable for particle rendering.

    // A system must be present.
    if (_system == nullptr) {
        throw std::runtime_error("ParticleLayer: system must not be null.");
    }

    // The system must expose a valid fluid object because visualization capacity
    // and particle storage are sourced from it.
    if (_system->fluid() == nullptr) {
        throw std::runtime_error("ParticleLayer: system fluid must not be null.");
    }
}

template <typename T>
GLenum
ParticleLayer<T>::vertex_component_type() const noexcept {
    // Return the OpenGL scalar component type corresponding to the particle
    // position component type T.
    if constexpr (std::is_same_v<T, double>) {
        // Use double-precision vertex components when T is double.
        return GL_DOUBLE;
    }

    // Default to single-precision vertex components for all other supported types.
    return GL_FLOAT;
}

template <typename T>
typename ParticleLayer<T>::Builder&
ParticleLayer<T>::Builder::with_color(const Vector4<T>& color) noexcept {
    // Stage particle render color.
    _color = color;
    return *this;
}

template <typename T>
typename ParticleLayer<T>::Builder&
ParticleLayer<T>::Builder::with_point_size(const T point_size) noexcept {
    _point_size = point_size;
    return *this;
}

template <typename T>
typename ParticleLayer<T>::Builder&
ParticleLayer<T>::Builder::with_system(const atlas::SystemHostPtr<T>& system) noexcept {
    // Stage the simulation system that provides particle data.
    _system = system;
    return *this;
}

template <typename T>
ParticleLayer<T>
ParticleLayer<T>::Builder::build() const {
    // Validate staged inputs before constructing the final layer by value.
    validate();
    return ParticleLayer<T>(_system, _color, _point_size);
}

template <typename T>
std::shared_ptr<ParticleLayer<T>>
ParticleLayer<T>::Builder::make_shared() const {
    // Validate staged inputs before constructing the final layer in shared storage.
    validate();
    return std::make_shared<ParticleLayer<T>>(_system, _color, _point_size);
}

template <typename T>
void
ParticleLayer<T>::Builder::validate() const {
    // Validate builder-side particle layer configuration.

    // A system must be provided.
    if (_system == nullptr) {
        throw std::runtime_error("ParticleLayer::Builder: system must not be null.");
    }

    // The system must expose a valid fluid object.
    if (_system->fluid() == nullptr) {
        throw std::runtime_error("ParticleLayer::Builder: system fluid must not be null.");
    }

    if (!std::isfinite(_point_size) || _point_size <= T(0)) {
        throw std::runtime_error("ParticleLayer::Builder: point_size must be finite and positive.");
    }
}

} // namespace atlas::vizkit

#endif
