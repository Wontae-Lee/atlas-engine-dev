#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file viewer.h
 * @brief Declares the top-level Vizkit viewer responsible for window creation, layer orchestration, and render-loop execution.
 *
 * @details
 * This header defines @ref atlas::vizkit::Viewer, a high-level visualization
 * entry point used to display Atlas simulation state through a GLFW/OpenGL-based
 * interactive window.
 *
 * The viewer is the central object coordinating:
 * - an optional bound @ref atlas::System,
 * - a GLFW window and OpenGL context,
 * - an interactive @ref atlas::vizkit::Camera,
 * - a list of render/update @ref atlas::vizkit::Layer instances,
 * - the initialization, runtime, and shutdown phases of visualization.
 *
 * ## Purpose
 * The viewer provides a simple runtime shell around Vizkit layers so users can:
 * - visualize simulation particles and geometry,
 * - attach multiple rendering layers to the same scene,
 * - run an interactive render loop with camera controls,
 * - couple visualization directly to a simulation system.
 *
 * ## Responsibilities
 * A viewer is responsible for:
 * - creating and owning the visualization window,
 * - initializing the OpenGL environment,
 * - initializing all registered layers,
 * - executing the per-frame update/render loop,
 * - shutting down layers and releasing graphics resources.
 *
 * ## Layer model
 * Layers are stored as shared pointers to the abstract @ref Layer interface.
 * This allows the viewer to host a heterogeneous set of rendering components,
 * such as:
 * - particle layers,
 * - geometry layers,
 * - custom overlays,
 * - future visualization widgets.
 *
 * Each registered layer participates in the viewer lifecycle through:
 * - initialization,
 * - per-frame update,
 * - shutdown.
 *
 * ## System integration
 * The viewer may be bound to an Atlas @ref System. This allows layers, such as
 * @ref atlas::vizkit::ParticleLayer, to read runtime simulation state directly.
 *
 * The viewer itself does not implement simulation logic. It acts as a rendering
 * host and scene coordinator.
 *
 * ## Window behavior
 * The viewer supports configuration of:
 * - window width,
 * - window height,
 * - window title,
 * - fullscreen mode.
 *
 * These values are staged either through the constructor or the nested
 * @ref Builder.
 *
 * ## Runtime flow
 * A typical run sequence is:
 * 1. construct the viewer,
 * 2. register one or more layers,
 * 3. call @ref run,
 * 4. allow the viewer to:
 *    - initialize GL,
 *    - initialize layers,
 *    - execute the main loop,
 *    - shut down layers,
 *    - clean up the GL context and window.
 *
 * ## Camera ownership
 * The viewer owns one @ref Camera instance that is shared with all layers during
 * initialization and per-frame updates.
 *
 * ## Builder pattern
 * The nested @ref Builder offers a fluent API for configuring the viewer before
 * constructing it by value or shared ownership.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the associated simulation system and layers.
 */

#include <atlas/system/system.h>
#include <atlas/math/vector/vector4.h>
#include <vizkit/camera/camera.h>
#include <vizkit/layer/layer.h>
#include <vizkit/macros/macros.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

/**
 * @brief Top-level visualization host for Atlas systems and Vizkit layers.
 *
 * @details
 * @ref Viewer is the primary Vizkit runtime object responsible for hosting an
 * interactive visualization session.
 *
 * It combines:
 * - a simulation-system handle,
 * - a window/context configuration,
 * - an interactive camera,
 * - a collection of rendering layers.
 *
 * ## High-level behavior
 * The viewer owns the render-loop lifecycle:
 * - initialize window and GL state,
 * - initialize attached layers,
 * - update the camera and layers every frame,
 * - shut everything down in reverse order.
 *
 * ## Design intent
 * The class is designed to be:
 * - easy to construct from a simulation system,
 * - extensible through user-provided layers,
 * - lightweight enough for examples and debugging tools,
 * - explicit about ownership of the visualization runtime.
 *
 * ## Layer orchestration
 * Layers are updated in registration order. Each layer receives:
 * - the active GLFW window,
 * - the active camera,
 * - the current timestep or frame delta.
 *
 * This makes the viewer a straightforward scene graph host without imposing a
 * more complex rendering architecture.
 *
 * ## Threading assumptions
 * The viewer is intended to be used from the host thread that owns the GLFW/OpenGL
 * context. Initialization, main-loop execution, and cleanup are therefore all
 * host-side operations.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the bound system and layers.
 */
template <typename T>
class Viewer final {
public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Viewer.
     *
     * @details
     * The builder stages:
     * - the bound system,
     * - window size,
     * - title,
     * - fullscreen mode,
     * and constructs either:
     * - a viewer by value, or
     * - a shared pointer to a viewer.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a viewer with default-initialized configuration:
     * - no bound system,
     * - zero width and height,
     * - default title,
     * - windowed mode,
     * - no created GLFW window,
     * - no registered layers.
     *
     * A default-constructed viewer typically requires further configuration
     * before @ref run can succeed.
     */
    Viewer() = default;

    /**
     * @brief Construct a viewer with explicit runtime configuration.
     *
     * @param system Host-side shared pointer to the bound simulation system.
     * @param width Window width in pixels.
     * @param height Window height in pixels.
     * @param title Window title string.
     * @param fullscreen Whether the viewer should open in fullscreen mode.
     * @param background_color RGBA color used when clearing the frame buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Viewer(SystemHostPtr<T> system,
           int width,
           int height,
           const char* title,
           bool fullscreen,
           const Vector4<T>& background_color = Vector4<T>(T(0.08), T(0.09), T(0.12), T(1))) noexcept;

    /**
     * @brief Destructor.
     *
     * @details
     * The destructor releases any remaining owned runtime resources according to
     * the implementation in `viewer.hpp`. This typically complements the cleanup
     * work done by @ref run and internal shutdown helpers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE ~Viewer();

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized @ref Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Register a visualization layer with the viewer.
     *
     * @details
     * The supplied layer is appended to the viewer's internal layer list and
     * will subsequently participate in:
     * - layer initialization,
     * - per-frame update,
     * - layer shutdown.
     *
     * Layers are processed in insertion order.
     *
     * @param layer Shared pointer to the layer to add.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_layer(const std::shared_ptr<Layer<T>>& layer);

    /**
     * @brief Return the bound simulation system.
     *
     * @return Const reference to the stored system handle.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SystemHostPtr<T>&
    system() const noexcept;

    /**
     * @brief Return mutable access to the viewer camera.
     *
     * @return Reference to the stored camera.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Camera&
    camera() noexcept;

    /**
     * @brief Execute the full viewer lifecycle.
     *
     * @details
     * A typical implementation performs:
     * 1. OpenGL/window initialization,
     * 2. layer initialization,
     * 3. execution of the main event/render loop,
     * 4. layer shutdown,
     * 5. graphics/window cleanup.
     *
     * The returned integer is typically used as a process-style status code.
     *
     * @return Exit status of the viewer session.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE int
    run();

private:
    /**
     * @brief Initialize the OpenGL context and window state.
     *
     * @details
     * This helper is responsible for the low-level graphics bootstrap phase,
     * including tasks such as:
     * - GLFW initialization,
     * - window creation,
     * - context activation,
     * - GLEW/OpenGL setup,
     * - basic render-state configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_gl();

    /**
     * @brief Initialize all registered layers.
     *
     * @details
     * Each layer is initialized with the active GLFW window and the viewer-owned
     * camera. This step is typically performed after the OpenGL context has been
     * successfully created.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_layers();

    /**
     * @brief Execute the main viewer loop.
     *
     * @details
     * A typical implementation repeatedly:
     * - polls window/input events,
     * - updates the camera,
     * - updates all registered layers,
     * - presents the rendered frame,
     * until the viewer window is closed.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    main_loop();

    /**
     * @brief Shut down all registered layers.
     *
     * @details
     * Invokes the shutdown path of each layer so GPU resources and other owned
     * visualization state can be released in an orderly manner.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    shutdown_layers();

    /**
     * @brief Clean up the OpenGL context and window resources.
     *
     * @details
     * This helper performs the final graphics teardown phase, such as:
     * - destroying the GLFW window,
     * - releasing the active context,
     * - terminating GLFW or related resources as appropriate.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    cleanup_gl();

private:
    /**
     * @brief Bound simulation system.
     *
     * @details
     * This system acts as the canonical runtime state source for visualization
     * layers that require direct simulation access.
     */
    SystemHostPtr<T> _system;

    /**
     * @brief Requested window width in pixels.
     */
    int _width = 0;

    /**
     * @brief Requested window height in pixels.
     */
    int _height = 0;

    /**
     * @brief Requested window title.
     *
     * @details
     * The pointed-to string is assumed to outlive viewer construction and use,
     * according to the implementation policy.
     */
    const char* _title = "Atlas Viewer";

    /**
     * @brief Whether fullscreen mode is requested.
     */
    bool _fullscreen = false;

    /**
     * @brief RGBA color used to clear the framebuffer before rendering each frame.
     */
    Vector4<T> _background_color { T(0.08), T(0.09), T(0.12), T(1) };

    /**
     * @brief Owned GLFW window handle.
     *
     * @details
     * This pointer remains null until the window/context is created.
     */
    GLFWwindow* _win = nullptr;

    /**
     * @brief Viewer-owned interactive camera.
     *
     * @details
     * Shared with all layers during initialization and per-frame updates.
     */
    Camera _cam {};

    /**
     * @brief Registered visualization layers.
     *
     * @details
     * Layers are processed in insertion order during initialization, updates,
     * and shutdown.
     */
    std::vector<std::shared_ptr<Layer<T>>> _layers;
};

/**
 * @brief Fluent builder for @ref Viewer.
 *
 * @details
 * The builder provides a controlled construction path for the viewer by staging:
 * - the bound simulation system,
 * - the window title,
 * - the window size,
 * - the fullscreen flag.
 *
 * ## Typical usage
 * @code
 * auto viewer = atlas::vizkit::Viewer<float>::builder()
 *     .with_system(system)
 *     .with_title("Atlas Viewer")
 *     .with_size(1280, 720)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_shared. Typical checks may include:
 * - a valid system handle is present if required,
 * - window dimensions are positive,
 * - title is non-null when required by implementation policy.
 *
 * The exact validation behavior is implementation-defined in `viewer.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the associated system and layers.
 */
template <typename T>
class Viewer<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the builder with:
     * - no bound system,
     * - zero width and height,
     * - the default title `"Atlas Viewer"`,
     * - windowed mode.
     */
    Builder() = default;

    /**
     * @brief Set the bound simulation system.
     *
     * @param system Host-side shared pointer to the simulation system.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_system(const SystemHostPtr<T>& system);

    /**
     * @brief Set the viewer window title.
     *
     * @param title Null-terminated window-title string.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_title(const char* title) noexcept;

    /**
     * @brief Set the viewer window size.
     *
     * @param width Requested window width in pixels.
     * @param height Requested window height in pixels.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_size(int width, int height) noexcept;

    /**
     * @brief Enable or disable fullscreen mode.
     *
     * @param fullscreen Fullscreen flag. Defaults to `true`.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fullscreen(bool fullscreen = true) noexcept;

    /**
     * @brief Set the viewer background clear color.
     *
     * @param color RGBA color used when clearing the frame buffer.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_background_color(const Vector4<T>& color) noexcept;

    /**
     * @brief Build a configured @ref Viewer by value after validation.
     *
     * @return Constructed viewer instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Viewer<T>
    build() const;

    /**
     * @brief Build a configured @ref Viewer in shared ownership after validation.
     *
     * @return `std::shared_ptr<Viewer<T>>` owning the constructed viewer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE std::shared_ptr<Viewer<T>>
    make_shared() const;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the staged system handle, window size,
     * title, and fullscreen configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending simulation-system dependency.
     */
    SystemHostPtr<T> _system;

    /**
     * @brief Pending window width.
     */
    int _width = 0;

    /**
     * @brief Pending window height.
     */
    int _height = 0;

    /**
     * @brief Pending window title.
     */
    const char* _title = "Atlas Viewer";

    /**
     * @brief Pending fullscreen flag.
     */
    bool _fullscreen = false;

    /**
     * @brief Pending background clear color.
     */
    Vector4<T> _background_color { T(0.08), T(0.09), T(0.12), T(1) };
};

} // namespace atlas::vizkit

#include <vizkit/viewer/viewer.hpp>

#endif
