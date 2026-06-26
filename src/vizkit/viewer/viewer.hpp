#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <string>
#include <utility>

namespace atlas::vizkit {

template <typename T>
Viewer<T>::Viewer(SystemHostPtr<T> system,
                  int width,
                  int height,
                  const char* title,
                  bool fullscreen,
                  const Vector4<T>& background_color,
                  std::optional<std::size_t> timestep_count) noexcept
    : _system(std::move(system))
    , _width(width)
    , _height(height)
    , _title(title ? title : "Atlas Viewer")
    , _fullscreen(fullscreen)
    , _background_color(background_color)
    , _timestep_count(timestep_count) {
    // Construct a viewer with its initial runtime configuration.
    //
    // Stored configuration:
    // - _system     : simulation system advanced during the render loop
    // - _width      : requested window width
    // - _height     : requested window height
    // - _title      : window title, defaulting to "Atlas Viewer" when null
    // - _fullscreen : whether the viewer opens in fullscreen mode
    // - _background_color : framebuffer clear color
    // - _timestep_count : optional number of simulation updates before exit
    //
    // This constructor only stores configuration.
    // OpenGL / GLFW resources are created later in init_gl().
}

template <typename T>
Viewer<T>::~Viewer() {
    // Ensure graphics resources are released when the viewer is destroyed.
    //
    // cleanup_gl() is written to be safe to call even if:
    // - initialization never completed
    // - resources were already cleaned up earlier
    cleanup_gl();
}

template <typename T>
typename Viewer<T>::Builder
Viewer<T>::builder() noexcept {
    // Return a fresh builder for staged Viewer<T> construction.
    return Builder {};
}

template <typename T>
void
Viewer<T>::add_layer(const std::shared_ptr<Layer<T>>& layer) {
    // Append one render layer to the viewer.
    //
    // Layers are processed in insertion order for:
    // - initialization
    // - per-frame updates / drawing
    // - shutdown
    _layers.push_back(layer);
}

template <typename T>
const SystemHostPtr<T>&
Viewer<T>::system() const noexcept {
    // Return the simulation system bound to this viewer.
    return _system;
}

template <typename T>
Camera&
Viewer<T>::camera() noexcept {
    return _cam;
}

template <typename T>
int
Viewer<T>::run() {
    // Execute the full viewer lifecycle.
    //
    // High-level workflow:
    // 1. initialize GLFW / OpenGL context
    // 2. initialize all render layers
    // 3. enter the main render/update loop
    // 4. shut down layers
    // 5. clean up windowing / OpenGL resources
    //
    // Return value:
    // - 0 on success
    // - 1 if an exception escapes the lifecycle steps
    try {
        // Create the OpenGL context and window.
        init_gl();

        // Let each registered layer allocate its own resources.
        init_layers();

        // Run the frame loop until the window closes.
        main_loop();

        // Shut down all layers after the loop exits normally.
        shutdown_layers();

        // Release GLFW / window resources.
        cleanup_gl();

        return 0;
    } catch (const std::exception& e) {
        // Perform best-effort cleanup before returning failure.
        shutdown_layers();
        cleanup_gl();

        (void)e;
        return 1;
    }
}

template <typename T>
void
Viewer<T>::init_gl() {
    // Initialize GLFW, create the window, create the OpenGL context,
    // initialize GLEW, and configure baseline render state.

    // Initialize GLFW first.
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW.");
    }

    // Request an OpenGL 3.3 core profile context.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Query the primary monitor.
    //
    // This is needed for:
    // - fullscreen window creation
    // - default resolution fallback
    // - centering the window when windowed
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (!monitor) {
        glfwTerminate();
        throw std::runtime_error("Failed to get primary monitor.");
    }

    // Query the current video mode of the primary monitor.
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    if (!mode) {
        glfwTerminate();
        throw std::runtime_error("Failed to query primary monitor video mode.");
    }

    // If width/height were left unspecified or zero-like, adopt monitor resolution.
    if (_width <= 0) _width = mode->width;
    if (_height <= 0) _height = mode->height;

    // Create the GLFW window.
    //
    // Fullscreen behavior:
    // - if _fullscreen is true, attach the window to the primary monitor
    // - otherwise create a normal windowed context
    _win = glfwCreateWindow(
        _width,
        _height,
        _title,
        _fullscreen ? monitor : nullptr,
        nullptr);

    if (!_win) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window.");
    }

    // Make the newly created window/context current on this thread.
    glfwMakeContextCurrent(_win);

    // Initialize GLEW after the context is current.
    glewExperimental        = GL_TRUE;
    const GLenum glew_error = glewInit();
    if (glew_error != GLEW_OK) {
        // On failure, clean up the partially initialized windowing state.
        glfwDestroyWindow(_win);
        _win = nullptr;
        glfwTerminate();

        throw std::runtime_error(
            std::string("Failed to initialize GLEW: ")
            + reinterpret_cast<const char*>(glewGetErrorString(glew_error)));
    }

    // Clear the benign GL error sometimes produced by GLEW initialization.
    glGetError();

    // Enable depth testing so 3D layers render with proper depth ordering.
    glEnable(GL_DEPTH_TEST);

    // Set a default line width used by line-based geometry layers.
    glLineWidth(2.0f);

    if (!_fullscreen) {
        // Center the window on the primary monitor in windowed mode.
        const int x = (mode->width - _width) / 2;
        const int y = (mode->height - _height) / 2;
        glfwSetWindowPos(_win, x, y);
    }
}

template <typename T>
void
Viewer<T>::init_layers() {
    // Initialize all registered layers.
    //
    // Each layer receives:
    // - the GLFW window / OpenGL context
    // - the shared camera object
    for (const auto& layer : _layers) {
        if (layer) {
            layer->init(_win, _cam);
        }
    }
}

template <typename T>
void
Viewer<T>::main_loop() {
    // Run the main application loop until the window is requested to close.
    //
    // Per-frame workflow:
    // 1. process OS / input events
    // 2. handle viewer-close shortcut
    // 3. update camera controls
    // 4. advance the simulation system when timestep budget remains
    // 5. refresh viewport and clear frame buffers
    // 6. update and draw each layer from the current simulation state
    // 7. present the frame
    std::size_t timesteps = 0;
    while (!glfwWindowShouldClose(_win)) {
        // Pump pending input and window events.
        glfwPollEvents();

        // Close the viewer when Escape is pressed.
        if (glfwGetKey(_win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(_win, GLFW_TRUE);
        }

        // Update camera state from current input.
        _cam.handle(_win);

        const bool should_update = !_timestep_count || timesteps < *_timestep_count;

        if (should_update && _system) {
            // Advance the simulation one step before rendering layers.
            _system->update();
            ++timesteps;
        }

        // Query current framebuffer size for HiDPI-aware viewport setup.
        int fbw = 1;
        int fbh = 1;
        glfwGetFramebufferSize(_win, &fbw, &fbh);

        // Match the OpenGL viewport to the framebuffer dimensions.
        glViewport(0, 0, fbw, fbh);

        // Clear the color and depth buffers for a new frame.
        glClearColor(
            static_cast<GLfloat>(_background_color.x),
            static_cast<GLfloat>(_background_color.y),
            static_cast<GLfloat>(_background_color.z),
            static_cast<GLfloat>(_background_color.w));
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Update and draw each registered layer.
        //
        // dt is taken from the bound simulation system.
        for (const auto& layer : _layers) {
            if (layer) {
                layer->update(_win, _cam, _system->dt());
            }
        }

        // Present the rendered frame.
        glfwSwapBuffers(_win);
    }
}

template <typename T>
void
Viewer<T>::shutdown_layers() {
    // Shut down all registered layers.
    //
    // Each layer gets a chance to release:
    // - GPU resources
    // - staging buffers
    // - any other per-layer state
    for (const auto& layer : _layers) {
        if (layer) {
            layer->shutdown();
        }
    }
}

template <typename T>
void
Viewer<T>::cleanup_gl() {
    // Release viewer-owned GLFW / window resources.
    //
    // Safe to call multiple times because:
    // - window destruction is guarded by _win
    // - glfwTerminate() is allowed as final cleanup
    if (_win) {
        glfwDestroyWindow(_win);
        _win = nullptr;
    }

    glfwTerminate();
}

template <typename T>
typename Viewer<T>::Builder&
Viewer<T>::Builder::with_system(const SystemHostPtr<T>& system) {
    // Stage the simulation system used by the viewer.
    //
    // The viewer requires a non-null system because:
    // - the main loop advances it
    // - layers may depend on it for rendering data
    if (!system) {
        throw std::runtime_error("Viewer::Builder: system must not be null.");
    }

    _system = system;
    return *this;
}

template <typename T>
typename Viewer<T>::Builder&
Viewer<T>::Builder::with_title(const char* title) noexcept {
    // Stage the window title.
    //
    // Null titles are normalized to the default viewer title.
    _title = title ? title : "Atlas Viewer";
    return *this;
}

template <typename T>
typename Viewer<T>::Builder&
Viewer<T>::Builder::with_size(int width, int height) noexcept {
    // Stage requested window dimensions.
    //
    // Interpretation:
    // - positive values -> explicit requested size
    // - zero values     -> later replaced with monitor dimensions in init_gl()
    _width  = width;
    _height = height;
    return *this;
}

template <typename T>
typename Viewer<T>::Builder&
Viewer<T>::Builder::with_fullscreen(bool fullscreen) noexcept {
    // Stage fullscreen/windowed mode selection.
    _fullscreen = fullscreen;
    return *this;
}

template <typename T>
typename Viewer<T>::Builder&
Viewer<T>::Builder::with_background_color(const Vector4<T>& color) noexcept {
    // Stage the framebuffer clear color used by the render loop.
    _background_color = color;
    return *this;
}

template <typename T>
typename Viewer<T>::Builder&
Viewer<T>::Builder::with_timestep_count(std::optional<std::size_t> timestep_count) noexcept {
    // Stage an optional finite simulation length.
    _timestep_count = timestep_count;
    return *this;
}

template <typename T>
void
Viewer<T>::Builder::validate() const {
    // Validate all required viewer configuration before construction.

    // A viewer requires a simulation system.
    if (_system == nullptr) {
        throw std::runtime_error("Viewer::Builder: system must not be null.");
    }

    // Width and height must not be negative.
    //
    // Zero is allowed and interpreted later as "use monitor default".
    if ((_width < 0) || (_height < 0)) {
        throw std::runtime_error("Viewer::Builder: width/height must be >= 0.");
    }

    // Title pointer must be valid after normalization.
    if (_title == nullptr) {
        throw std::runtime_error("Viewer::Builder: title must not be null.");
    }
}

template <typename T>
Viewer<T>
Viewer<T>::Builder::build() const {
    // Validate builder state before constructing the viewer by value.
    validate();
    return Viewer<T>(_system, _width, _height, _title, _fullscreen, _background_color, _timestep_count);
}

template <typename T>
std::shared_ptr<Viewer<T>>
Viewer<T>::Builder::make_shared() const {
    // Build the viewer by value, then move it into shared storage.
    auto viewer = build();
    return std::make_shared<Viewer<T>>(std::move(viewer));
}

} // namespace atlas::vizkit

#endif
