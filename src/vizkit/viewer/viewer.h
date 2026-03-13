#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/camera/camera.h>
#include <vizkit/layer/layer.h>
#include <vizkit/macros/macros.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

/**
 * @brief Interactive VizKit viewer that owns a window, camera, and layer stack.
 *
 * @tparam T Scalar type used for frame delta time and camera/layer update flow.
 *
 * `Viewer` is the top-level runtime object for VizKit visualization. It is
 * responsible for:
 * - creating and destroying the GLFW window and OpenGL context
 * - initializing GLEW and core GL state
 * - storing a shared orbit camera
 * - driving a per-frame main loop
 * - dispatching lifecycle callbacks to registered layers
 *
 * Execution model:
 * 1. `run()` initializes the graphics subsystem.
 * 2. All registered layers receive @ref Layer::init.
 * 3. The viewer enters its event/render loop.
 * 4. Each frame it processes input, updates the camera, clears the framebuffer,
 *    and calls @ref Layer::update on every layer.
 * 5. On exit or failure, layers are shut down and GL resources are cleaned up.
 *
 * The viewer stores layers as shared pointers so multiple owners can retain
 * references to the same layer object outside the viewer if needed.
 */
template <typename T>
class Viewer final {
public:
    /** @brief Fluent builder for constructing @ref Viewer instances. */
    class Builder;

public:
    /** @brief Constructs a viewer with default configuration values. */
    Viewer() = default;

    /**
     * @brief Constructs a viewer from explicit runtime settings.
     *
     * @param dt Fixed frame delta time passed to layers each update.
     * @param width Requested window width in pixels. Non-positive values are
     * resolved later from the primary monitor mode.
     * @param height Requested window height in pixels. Non-positive values are
     * resolved later from the primary monitor mode.
     * @param title Window title string. A null pointer is replaced with the
     * default title `"Atlas Viewer"`.
     * @param fullscreen Whether to create the window in fullscreen mode.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Viewer(T dt,
           int width,
           int height,
           const char* title,
           bool fullscreen) noexcept;

    /**
     * @brief Destroys the viewer and releases window/context resources.
     *
     * The destructor calls the internal GL cleanup routine.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE ~Viewer();

    /**
     * @brief Creates a builder for @ref Viewer.
     *
     * @return Default-initialized builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Adds one layer to the viewer's render/update stack.
     *
     * @param layer Shared pointer to a layer instance.
     *
     * Layers are processed in insertion order for both initialization and
     * per-frame updates.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_layer(const std::shared_ptr<Layer<T>>& layer);

    /**
     * @brief Runs the complete viewer lifecycle.
     *
     * @return `0` on success, `1` if initialization or execution throws.
     *
     * `run()` performs:
     * - OpenGL/GLFW initialization
     * - layer initialization
     * - the main render loop
     * - orderly layer shutdown and GL cleanup
     *
     * Any thrown exception is caught, logged, and converted into a non-zero
     * return code after cleanup is attempted.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE int
    run();

private:
    /**
     * @brief Initializes GLFW, creates the window, and configures GL state.
     *
     * @throws std::runtime_error If GLFW, GLEW, monitor query, or window
     * creation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_gl();

    /**
     * @brief Calls `init()` on every registered layer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_layers();

    /**
     * @brief Executes the event/render loop until the window should close.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    main_loop();

    /**
     * @brief Calls `shutdown()` on every registered layer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    shutdown_layers();

    /**
     * @brief Destroys the window and terminates GLFW.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    cleanup_gl();

private:
    /** @brief Fixed delta time passed to layers every frame. */
    T _dt              = static_cast<T>(0.01);
    /** @brief Requested window width in pixels. */
    int _width         = 0;
    /** @brief Requested window height in pixels. */
    int _height        = 0;
    /** @brief Window title shown by the platform window manager. */
    const char* _title = "Atlas Viewer";
    /** @brief Whether the viewer should create a fullscreen window. */
    bool _fullscreen   = false;

    /** @brief Owned GLFW window handle, or `nullptr` when not initialized. */
    GLFWwindow* _win = nullptr;
    /** @brief Shared orbit camera used by all layers. */
    Camera _cam {};
    /** @brief Ordered collection of layers managed by the viewer. */
    std::vector<std::shared_ptr<Layer<T>>> _layers;
};

/* ====================================================================== */
/* Builder                                                                 */
/* ====================================================================== */

template <typename T>
class Viewer<T>::Builder final {
public:
    /** @brief Creates a builder with default viewer settings. */
    Builder() = default;

    /**
     * @brief Sets the fixed frame delta time.
     *
     * @param dt Positive delta time value supplied to layers.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_dt(T dt) noexcept;

    /**
     * @brief Sets the viewer window title.
     *
     * @param title Title string. Null is normalized to the default title.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_title(const char* title) noexcept;

    /**
     * @brief Sets the requested window size.
     *
     * @param width Requested width in pixels.
     * @param height Requested height in pixels.
     * @return Reference to this builder.
     *
     * Zero values are allowed and later interpreted by the viewer as a request
     * to use the current monitor resolution.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_size(int width, int height) noexcept;

    /**
     * @brief Enables or disables fullscreen mode.
     *
     * @param fullscreen Desired fullscreen flag.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fullscreen(bool fullscreen = true) noexcept;

    /**
     * @brief Builds a @ref Viewer by value.
     *
     * @return Constructed viewer.
     *
     * @throws std::runtime_error If builder validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Viewer<T>
    build() const;

    /**
     * @brief Builds a heap-allocated @ref Viewer.
     *
     * @return Shared pointer owning the constructed viewer.
     *
     * @throws std::runtime_error If builder validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE std::shared_ptr<Viewer<T>>
    make_shared() const;

private:
    /**
     * @brief Validates the builder configuration.
     *
     * Validation rules:
     * - `dt` must be strictly positive
     * - `width` and `height` must be non-negative
     * - `title` must not be null
     *
     * @throws std::runtime_error If any rule is violated.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /** @brief Fixed delta time to apply to the constructed viewer. */
    T _dt              = static_cast<T>(0.01);
    /** @brief Requested window width in pixels. */
    int _width         = 0;
    /** @brief Requested window height in pixels. */
    int _height        = 0;
    /** @brief Requested viewer title. */
    const char* _title = "Atlas Viewer";
    /** @brief Requested fullscreen flag. */
    bool _fullscreen   = false;
};

} // namespace atlas::vizkit

#include <vizkit/viewer/viewer.hpp>

#endif
