#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/macros/macros.h>
#include <vizkit/camera/camera.h>
namespace atlas::vizkit {

/**
 * @brief Abstract interface for one render/update layer in a VizKit viewer.
 *
 * @tparam T Scalar type used for frame delta time.
 *
 * A layer is a self-contained rendering or visualization module that
 * participates in the viewer lifecycle:
 * 1. @ref init is called once after the OpenGL context has been created.
 * 2. @ref update is called once per frame while the viewer main loop runs.
 * 3. @ref shutdown is called before viewer teardown to release owned resources.
 *
 * Layers receive:
 * - the active GLFW window for window/context-dependent operations
 * - the shared camera used by the viewer
 * - the per-frame delta time during updates
 *
 * Typical layer responsibilities include:
 * - creating GPU resources during initialization
 * - reacting to camera state or simulation updates each frame
 * - drawing geometry or overlays
 * - releasing resources during shutdown
 */
template <typename T>
class Layer {
public:
    virtual ~Layer() = default;

    /**
     * @brief Initializes the layer after the viewer has created the GL context.
     *
     * @param window Active GLFW window associated with the viewer.
     * @param camera Shared viewer camera available during initialization.
     *
     * Implementations typically allocate GPU resources, compile shaders, or
     * prepare CPU-side state needed for later rendering.
     */
    virtual void
    init(GLFWwindow* window, Camera& camera)
        = 0;

    /**
     * @brief Advances and renders the layer for one frame.
     *
     * @param window Active GLFW window associated with the viewer.
     * @param camera Shared viewer camera reflecting current user input.
     * @param dt Frame delta time supplied by the viewer.
     *
     * Implementations may update simulation state, synchronize render data,
     * upload modified buffers, and issue OpenGL draw calls.
     */
    virtual void
    update(GLFWwindow* window, Camera& camera, T dt)
        = 0;

    /**
     * @brief Releases resources owned by the layer.
     *
     * The default implementation does nothing, allowing lightweight layers to
     * omit custom teardown when they do not manage explicit resources.
     */
    virtual void
    shutdown() { }
};

}

#endif
