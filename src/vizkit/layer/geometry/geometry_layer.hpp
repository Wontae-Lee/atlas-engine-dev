#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <cmath>
#include <vizkit/shader/glsl.h>

namespace atlas::vizkit {

template <typename T>
GeometryLayer<T>::GeometryLayer(unsigned int primitive_mode,
                                const atlas::UnitHostPtr<T>& unit)
    : _unit(unit)
    , _primitive_mode(primitive_mode) {
    // Construct a generic geometry visualization layer.
    //
    // Parameters:
    // - primitive_mode : OpenGL primitive topology used for rendering
    //   (for example GL_LINES or GL_TRIANGLES)
    // - unit           : source unit that provides:
    //   * geometry definition
    //   * transform updates through sync_operator()
    //
    // The constructor only stores configuration.
    // Actual geometry extraction and GPU resource creation happen in init().
}

template <typename T>
void
GeometryLayer<T>::init(GLFWwindow* window, Camera& camera) {
    // Initialize CPU-side geometry buffers and GPU-side rendering resources.
    //
    // High-level workflow:
    // 1. build local-space geometry vertices
    // 2. initialize world-space vertex buffer from local-space data
    // 3. synchronize vertices through the unit transform
    // 4. create shader program
    // 5. create VAO/VBO and upload initial vertex data
    // 6. cache uniform locations used during rendering

    // This layer does not currently use the window or camera during initialization.
    (void)window;
    (void)camera;

    // Rebuild the local-space geometry from scratch.
    //
    // build_geometry() is implemented by derived layers such as:
    // - BoxLayer
    // - CircleLayer
    // - CylinderLayer
    // - future square-like geometry layers
    //
    // The result is stored in _local_positions.
    _local_positions.clear();
    build_geometry(_local_positions);

    // Start world-space positions as a direct copy of local-space positions.
    //
    // This gives synchronize() a correctly sized destination buffer and provides
    // a reasonable initial state even before any unit transform is applied.
    _world_positions = _local_positions;

    // Apply the current unit transform once so the initial uploaded vertex buffer
    // is already in world space.
    //
    // Passing dt = 0 indicates that synchronization here is purely geometric,
    // not time integration.
    synchronize(_world_positions, T(0));

    // Cache vertex count for later draw calls.
    _vertex_count = static_cast<int>(_world_positions.size());

    // If there is no geometry to draw, stop initialization early.
    //
    // In this case:
    // - no shader is created
    // - no VAO/VBO is allocated
    // - update() will later early-return because _vertex_count <= 0
    if (_vertex_count <= 0) return;

    // Create the shader program used by this geometry layer.
    //
    // Current shader choice:
    // - k_line_vs : vertex shader source
    // - k_line_fs : fragment shader source
    //
    // Even triangle-based layers currently reuse this generic shader pair.
    _program = std::make_unique<ShaderProgram>(k_line_vs, k_line_fs);

    // Allocate one vertex array object and one vertex buffer object.
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);

    // Bind the VAO so subsequent vertex-format state is recorded into it.
    glBindVertexArray(_vao);

    // Bind the VBO so vertex data can be uploaded.
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);

    // Upload initial world-space vertex data to GPU memory.
    //
    // Buffer usage hint:
    // - GL_DYNAMIC_DRAW
    //
    // because geometry positions may be updated each frame by synchronize().
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(_world_positions.size() * sizeof(Vector3<T>)),
        _world_positions.data(),
        GL_DYNAMIC_DRAW);

    // Enable vertex attribute slot 0.
    //
    // Convention used here:
    // - attribute 0 = position
    glEnableVertexAttribArray(0);

    // Describe the layout of one vertex position in the buffer.
    //
    // Layout:
    // - 3 components
    // - float type
    // - tightly packed as Vector3<T>
    //
    // Note:
    // - this assumes Vector3<T> is layout-compatible with 3 consecutive floats
    //   when used with the configured rendering path
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Vector3<T>),
        reinterpret_cast<void*>(0));

    // Unbind VAO to prevent accidental external state modification.
    glBindVertexArray(0);

    // Unbind VBO for the same reason.
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Cache uniform locations used every frame during rendering.
    //
    // "MVP"   : combined model-view-projection matrix
    // "uColor": uniform RGBA draw color
    _u_mvp   = _program->uniform_loc("MVP");
    _u_color = _program->uniform_loc("uColor");
}

template <typename T>
void
GeometryLayer<T>::update(GLFWwindow* window, Camera& camera, T dt) {
    // Update the layer and issue one draw call.
    //
    // High-level workflow:
    // 1. early-out if rendering resources are not initialized
    // 2. update the bound unit transform
    // 3. resynchronize world-space vertices if transform changed them
    // 4. update the GPU vertex buffer when needed
    // 5. build MVP from framebuffer size and camera state
    // 6. bind shader, upload uniforms, and draw geometry

    // Rendering is impossible if:
    // - no shader program exists
    // - no VAO exists
    // - no vertices exist
    if (!_program || !_vao || _vertex_count <= 0) return;

    if (_unit) {
        // Advance the bound unit by dt before synchronizing geometry.
        //
        // This allows moving / rotating units to update the rendered world-space
        // geometry every frame.
        _unit->update(dt);
    }

    // Recompute world-space positions from local-space geometry and the current
    // unit transform. Upload only if any vertex changed by more than epsilon.
    if (synchronize(_world_positions, dt)) {
        // Bind the vertex buffer so its contents can be updated in-place.
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);

        // Replace the existing GPU buffer contents with the updated world-space
        // vertex array.
        //
        // glBufferSubData is sufficient because:
        // - buffer size is unchanged
        // - only contents are being refreshed
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            static_cast<GLsizeiptr>(_world_positions.size() * sizeof(Vector3<T>)),
            _world_positions.data());

        // Unbind the vertex buffer after update.
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // Query framebuffer size so the camera can build a correct aspect-aware MVP.
    int w = 1;
    int h = 1;
    glfwGetFramebufferSize(window, &w, &h);

    // Build the current model-view-projection matrix.
    float mvp[16];
    camera.build_mvp(w, h, mvp);

    // Activate the shader program before uploading uniforms and drawing.
    _program->use();

    // Upload the MVP matrix.
    glUniformMatrix4fv(_u_mvp, 1, GL_FALSE, mvp);

    // Upload a default geometry color if the shader exposes the uniform.
    if (_u_color >= 0) {
        glUniform4f(_u_color, _color.x, _color.y, _color.z, _color.w);
    }

    // Allow geometry layers to use alpha in the shared fragment shader.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Bind the VAO containing vertex format + buffer binding state.
    glBindVertexArray(_vao);

    // Issue the draw call using the layer's configured primitive topology.
    glDrawArrays(_primitive_mode, 0, _vertex_count);

    // Unbind VAO after drawing.
    glBindVertexArray(0);

    glDisable(GL_BLEND);
}

template <typename T>
void
GeometryLayer<T>::set_color(const Vector4<T>& color) noexcept {
    _color = color;
}

template <typename T>
void
GeometryLayer<T>::shutdown() {
    // Release GPU resources and reset CPU-side cached state.
    //
    // After shutdown():
    // - the layer can no longer render until init() is called again
    // - all cached geometry/state is reset to an empty state

    if (_vbo) {
        // Delete the vertex buffer if it exists.
        glDeleteBuffers(1, &_vbo);
        _vbo = 0;
    }

    if (_vao) {
        // Delete the vertex array object if it exists.
        glDeleteVertexArrays(1, &_vao);
        _vao = 0;
    }

    // Reset draw-related cached values.
    _vertex_count = 0;
    _u_mvp        = -1;
    _u_color      = -1;

    // Release CPU-side geometry caches.
    _local_positions.clear();
    _world_positions.clear();

    // Destroy the shader program object.
    _program.reset();
}

template <typename T>
bool
GeometryLayer<T>::synchronize(std::vector<Vector3<T>>& world_positions, T dt) {
    // Synchronize world-space vertex positions with local geometry and current unit pose.
    //
    // Return value:
    // - true  -> at least one vertex changed enough that GPU upload is needed
    // - false -> no meaningful vertex change was detected
    //
    // Notes:
    // - dt is currently unused
    // - synchronization depends only on:
    //   * _local_positions
    //   * optional unit transform

    (void)dt;

    // Keep destination array size consistent with the source local geometry size.
    if (world_positions.size() != _local_positions.size()) {
        world_positions.resize(_local_positions.size());
    }

    // Track whether any vertex changed.
    bool changed = false;

    if (!_unit) {
        // No bound unit means there is no transform to apply.
        //
        // In this case, world positions are simply identical to local positions.
        for (std::size_t i = 0; i < _local_positions.size(); ++i) {
            const auto& src = _local_positions[i];
            auto& dst       = world_positions[i];

            // Detect whether the destination differs from the source by more than epsilon.
            if (std::abs(dst.x - src.x) > eps
                || std::abs(dst.y - src.y) > eps
                || std::abs(dst.z - src.z) > eps) {
                changed = true;
            }

            // Copy local-space position directly into world-space buffer.
            dst = src;
        }

        return changed;
    }

    // Read the unit's current synchronization operator.
    const auto& sync = _unit->sync_operator();

    for (std::size_t i = 0; i < _local_positions.size(); ++i) {
        // Transform the current local-space vertex into world space.
        const Vector3<T> transformed = sync.sync_to_world(_local_positions[i]);

        auto& dst = world_positions[i];

        // Detect whether this transformed vertex differs from the previously stored
        // world-space vertex by more than epsilon.
        if (std::abs(dst.x - transformed.x) > eps
            || std::abs(dst.y - transformed.y) > eps
            || std::abs(dst.z - transformed.z) > eps) {
            changed = true;
        }

        // Store the transformed world-space position.
        dst = transformed;
    }

    return changed;
}

} // namespace atlas::vizkit

#endif
