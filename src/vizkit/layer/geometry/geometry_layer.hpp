#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <cmath>
#include <vizkit/shader/glsl.h>

namespace atlas::vizkit {

template <typename T>
GeometryLayer<T>::GeometryLayer(unsigned int primitive_mode,
                                const atlas::UnitHostPtr<T>& unit)
    // The primitive mode determines how OpenGL groups the uploaded vertex list:
    // - GL_LINES     : pairs of vertices become independent segments
    // - GL_TRIANGLES : triples of vertices become independent triangles
    //
    // The base class stays agnostic to the shape; it only preserves the chosen
    // interpretation for the final glDrawArrays call.
    : _unit(unit)
    , _primitive_mode(primitive_mode) { }

template <typename T>
void
GeometryLayer<T>::init(GLFWwindow* window, Camera& camera) {
    (void)window;
    (void)camera;

    // Step 1: build local-space geometry.
    //
    // Derived classes generate vertices in their own natural coordinate frame.
    // Examples:
    // - a box layer emits the 12 box edges in local coordinates
    // - a cylinder layer tessellates a canonical cylinder around its local axis
    //
    // At this stage, the vertices describe the object itself, not yet its
    // placement in the world.
    _local_positions.clear();
    build_geometry(_local_positions);

    // Step 2: initialize the world-space buffer from the local vertex set.
    // The copy is intentional: _local_positions stays as the immutable template
    // geometry, while _world_positions becomes the mutable render buffer after
    // applying rigid transforms.
    _world_positions = _local_positions;

    // Step 3: synchronize local coordinates into world coordinates.
    //
    // Mathematically, if a unit has rigid transform (R, t), each point p_local
    // becomes:
    //   p_world = R * p_local + t
    //
    // This conversion is handled by the unit's SyncOperator.
    synchronize(_world_positions, T(0));

    // The GPU draw count is simply the size of the uploaded flat vertex array.
    // glDrawArrays does not know about higher-level shapes; it sees only a
    // linear list and groups vertices according to _primitive_mode.
    _vertex_count = static_cast<int>(_world_positions.size());
    if (_vertex_count <= 0) return;

    // Compile/link the shader program used by this layer.
    // The chosen GLSL pair consumes positions and an MVP matrix, then renders a
    // solid color primitive. All geometry layers share this simple path.
    _program = std::make_unique<ShaderProgram>(k_line_vs, k_line_fs);

    // Create the OpenGL objects that describe:
    // - where vertex data is stored (_vbo)
    // - how that data is interpreted during rendering (_vao)
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);

    // Bind the VAO first so subsequent vertex attribute state is recorded into
    // it. Then bind the VBO as the current GL_ARRAY_BUFFER target.
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);

    // Upload the entire world-space vertex array into GPU memory.
    //
    // GL_DYNAMIC_DRAW signals that:
    // - data will be supplied by the CPU
    // - data may change repeatedly
    // - data will be used many times for drawing
    //
    // This is appropriate because synchronize() may update the vertex positions
    // every frame when the bound unit moves.
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(_world_positions.size() * sizeof(Vector3<T>)),
        _world_positions.data(),
        GL_DYNAMIC_DRAW);

    // Enable vertex attribute 0 and describe its layout.
    //
    // The shader expects a vec3 position. Each vertex occupies sizeof(Vector3<T>)
    // bytes and starts immediately at offset 0. Because the struct stores x/y/z
    // contiguously, OpenGL can read the buffer directly as tightly packed 3D
    // coordinates.
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Vector3<T>),
        reinterpret_cast<void*>(0));

    // Unbind to leave the OpenGL state machine in a cleaner state for callers.
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Cache uniform locations once during initialization.
    // Looking them up every frame would be unnecessary driver overhead.
    _u_mvp   = _program->uniform_loc("MVP");
    _u_color = _program->uniform_loc("uColor");
}

template <typename T>
void
GeometryLayer<T>::update(GLFWwindow* window, Camera& camera, T dt) {
    // If initialization never produced drawable state, there is nothing to do.
    if (!_program || !_vao || _vertex_count <= 0) return;

    if (_unit) {
        // Advance the bound unit's simulation or transform state first.
        // This makes rendering observe the latest pose for the current frame.
        _unit->update(dt);
    }

    // Recompute world positions only if needed.
    //
    // synchronize() compares the new transformed points against the cached
    // buffer and returns true only when any point changed beyond a small
    // epsilon. This avoids redundant buffer uploads.
    if (synchronize(_world_positions, dt)) {
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);

        // Update the existing GPU allocation in place.
        //
        // glBufferSubData avoids reallocating storage; it only replaces the
        // contents of the already-created VBO. This is a common dynamic-geometry
        // pattern when topology is fixed but positions move over time.
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            static_cast<GLsizeiptr>(_world_positions.size() * sizeof(Vector3<T>)),
            _world_positions.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // Query the framebuffer size, not just the window size.
    // On high-DPI displays these can differ, and projection math must use the
    // actual render target dimensions to keep aspect ratio correct.
    int w = 1;
    int h = 1;
    glfwGetFramebufferSize(window, &w, &h);

    // Build the model-view-projection matrix.
    //
    // In this renderer, geometry is already in world space after synchronize(),
    // so the "model" part is effectively identity here. The camera builds the
    // combined View-Projection transform, and the shader applies:
    //   p_clip = MVP * p_world
    float mvp[16];
    camera.build_mvp(w, h, mvp);

    // Make the shader program current before setting uniforms or drawing.
    _program->use();

    // Upload the 4x4 transform. GL_FALSE means the matrix is already stored in
    // the memory layout expected by OpenGL, so no transpose is requested.
    glUniformMatrix4fv(_u_mvp, 1, GL_FALSE, mvp);

    if (_u_color >= 0) {
        // Supply a constant RGBA color for the whole draw call.
        // With this shader path there is no per-vertex color attribute; one
        // uniform controls the appearance of the entire layer.
        glUniform4f(_u_color, 0.90f, 0.95f, 1.00f, 1.00f);
    }

    // Draw the uploaded vertex stream.
    //
    // OpenGL interprets the same raw vertex array differently depending on
    // _primitive_mode:
    // - GL_LINES     : vertices (0,1), (2,3), ...
    // - GL_TRIANGLES : vertices (0,1,2), (3,4,5), ...
    glBindVertexArray(_vao);
    glDrawArrays(_primitive_mode, 0, _vertex_count);
    glBindVertexArray(0);
}

template <typename T>
void
GeometryLayer<T>::shutdown() {
    // Release GPU objects in reverse order of use. Zeroing the handles after
    // deletion prevents accidental double-destroy on repeated shutdown calls.
    if (_vbo) {
        glDeleteBuffers(1, &_vbo);
        _vbo = 0;
    }
    if (_vao) {
        glDeleteVertexArrays(1, &_vao);
        _vao = 0;
    }

    // Reset CPU-side cached state so the layer returns to a clean, empty state.
    _vertex_count = 0;
    _u_mvp        = -1;
    _u_color      = -1;
    _local_positions.clear();
    _world_positions.clear();
    _program.reset();
}

template <typename T>
bool
GeometryLayer<T>::synchronize(std::vector<Vector3<T>>& world_positions, T dt) {
    (void)dt;

    // Keep the destination array structurally aligned with the source geometry.
    // The base implementation assumes topology is fixed: only vertex positions
    // move, not the number of vertices.
    if (world_positions.size() != _local_positions.size()) {
        world_positions.resize(_local_positions.size());
    }

    // Two positions are considered identical if all coordinates differ by less
    // than eps. This suppresses tiny floating-point noise from triggering a GPU
    // buffer upload every frame.
    const T eps = static_cast<T>(1e-6);
    bool changed = false;

    if (!_unit) {
        // Without a unit there is no rigid transform, so local coordinates are
        // already the render coordinates. This branch degenerates to a copy.
        for (std::size_t i = 0; i < _local_positions.size(); ++i) {
            const auto& src = _local_positions[i];
            auto& dst       = world_positions[i];

            if (std::abs(dst.x - src.x) > eps || std::abs(dst.y - src.y) > eps || std::abs(dst.z - src.z) > eps) {
                changed = true;
            }

            dst = src;
        }
        return changed;
    }

    // Fetch the unit's rigid transform operator once.
    const auto& sync = _unit->sync_operator();

    for (std::size_t i = 0; i < _local_positions.size(); ++i) {
        // Treat every vertex as a point, not as a direction.
        //
        // For points, rigid synchronization applies both rotation and
        // translation:
        //   p_world = R * p_local + t
        const Vector3<T> transformed = sync.sync_to_world(_local_positions[i]);
        auto& dst                    = world_positions[i];

        if (std::abs(dst.x - transformed.x) > eps || std::abs(dst.y - transformed.y) > eps || std::abs(dst.z - transformed.z) > eps) {
            changed = true;
        }

        dst = transformed;
    }

    return changed;
}

} // namespace atlas::vizkit

#endif
