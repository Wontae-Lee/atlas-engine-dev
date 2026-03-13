#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <string>
#include <utility>

namespace atlas::vizkit {

/* ====================================================================== */
/* Viewer<T>                                                               */
/* ====================================================================== */

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
Viewer<T>::Viewer(T dt,
                  int width,
                  int height,
                  const char* title,
                  bool fullscreen) noexcept
    // Store viewer configuration only.
    // Actual window/context creation is deferred to run() -> init_gl().
    : _dt(dt)
    , _width(width)
    , _height(height)
    , _title(title ? title : "Atlas Viewer")
    , _fullscreen(fullscreen) { }

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    Viewer<T>::~Viewer() {
    // The destructor is a last-resort cleanup path. cleanup_gl() is also called
    // explicitly in run(), so this must remain safe when the window is already
    // null or GLFW has effectively been cleaned up.
    cleanup_gl();
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE typename Viewer<T>::Builder
Viewer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
Viewer<T>::add_layer(const std::shared_ptr<Layer<T>>& layer) {
    // Layers are stored in insertion order. That order defines:
    // - initialization order
    // - per-frame update/draw order
    // - shutdown order
    //
    // This matters visually because later layers may overdraw earlier ones.
    _layers.push_back(layer);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE int
Viewer<T>::run() {
    try {
        // Full viewer lifecycle:
        // 1. create the window and OpenGL context
        // 2. initialize every registered layer
        // 3. enter the event/render loop
        // 4. shut down layers
        // 5. destroy window/context state
        init_gl();
        init_layers();
        main_loop();
        shutdown_layers();
        cleanup_gl();
        return 0;
    } catch (const std::exception& e) {
        // Any exception from GLFW/GLEW setup, layer initialization, or per-frame
        // rendering is converted into a logged failure and a non-zero exit code.
        // Cleanup is still attempted so partially initialized resources do not
        // leak.
        atlas::logger::error() << "Viewer failed: " << e.what();
        shutdown_layers();
        cleanup_gl();
        return 1;
    }
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
Viewer<T>::init_gl() {
    // Initialize GLFW, which provides:
    // - window creation
    // - OpenGL context creation
    // - input/event handling
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW.");
    }

    // Request an OpenGL 3.3 Core profile context.
    //
    // "Core profile" removes deprecated fixed-function APIs and ensures the
    // viewer uses the programmable pipeline via shaders, VAOs, VBOs, uniforms,
    // and explicit draw calls.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Query the primary monitor so fullscreen creation and default sizing can
    // be derived from the monitor currently driving the display.
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (!monitor) {
        glfwTerminate();
        throw std::runtime_error("Failed to get primary monitor.");
    }

    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    if (!mode) {
        glfwTerminate();
        throw std::runtime_error("Failed to query primary monitor video mode.");
    }

    // If the builder left width/height unspecified (<= 0), fall back to the
    // monitor's current display mode. This gives a sensible default both for
    // fullscreen and for large windowed sessions.
    if (_width <= 0) _width = mode->width;
    if (_height <= 0) _height = mode->height;

    // Create the platform window and the associated OpenGL context.
    //
    // If fullscreen is requested, the monitor pointer is passed to GLFW so the
    // window becomes a fullscreen window on that monitor. Otherwise nullptr
    // creates a regular windowed mode surface.
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

    // Make the new OpenGL context current on this thread. All subsequent GLEW
    // and OpenGL calls depend on having a current context first.
    glfwMakeContextCurrent(_win);

    // GLEW loads OpenGL function pointers beyond the base platform API.
    //
    // glewExperimental = GL_TRUE is commonly used with core-profile contexts so
    // modern entry points are exposed consistently.
    glewExperimental        = GL_TRUE;
    const GLenum glew_error = glewInit();
    if (glew_error != GLEW_OK) {
        glfwDestroyWindow(_win);
        _win = nullptr;
        glfwTerminate();

        throw std::runtime_error(
            std::string("Failed to initialize GLEW: ") + reinterpret_cast<const char*>(glewGetErrorString(glew_error)));
    }

    // Some GLEW setups emit a benign GL error during initialization. Consume it
    // so later error checks are not polluted by stale state.
    glGetError();

    // Enable depth testing so fragments nearer to the camera occlude farther
    // fragments.
    //
    // Without depth testing, primitives would be composited only in draw order,
    // producing incorrect visibility for 3D geometry.
    glEnable(GL_DEPTH_TEST);

    // Set a default rasterization width for line primitives. This affects line-
    // based layers such as wireframe boxes.
    glLineWidth(2.0f);

    if (!_fullscreen) {
        // Center the window in windowed mode for a more predictable initial
        // presentation. Fullscreen windows are monitor-managed instead.
        const int x = (mode->width - _width) / 2;
        const int y = (mode->height - _height) / 2;
        glfwSetWindowPos(_win, x, y);
    }
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
Viewer<T>::init_layers() {
    // Forward initialization to every registered layer after the GL context is
    // ready. This ordering is essential because layers often allocate OpenGL
    // objects, which would fail before context creation.
    for (const auto& layer : _layers) {
        if (layer) {
            layer->init(_win, _cam);
        }
    }
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
Viewer<T>::main_loop() {
    // Classical event/render loop:
    // - process platform input/events
    // - update interactive camera state
    // - clear framebuffer
    // - update/draw each layer
    // - present the back buffer
    while (!glfwWindowShouldClose(_win)) {
        // Pump the OS event queue so keyboard, mouse, and window events are
        // delivered to GLFW's internal state.
        glfwPollEvents();

        // Provide a simple global quit shortcut.
        if (glfwGetKey(_win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(_win, GLFW_TRUE);
        }

        // Update orbit-camera parameters from the current input state.
        _cam.handle(_win);

        // Use framebuffer size rather than logical window size.
        // This matters on high-DPI displays where one screen-space window unit
        // can map to multiple physical pixels.
        int fbw = 1;
        int fbh = 1;
        glfwGetFramebufferSize(_win, &fbw, &fbh);

        // Define the viewport transform from normalized device coordinates
        // (roughly [-1,1]^2 after projection) into framebuffer pixel
        // coordinates.
        glViewport(0, 0, fbw, fbh);

        // Clear the color and depth buffers at the start of each frame.
        //
        // - color clear establishes the background color
        // - depth clear resets per-pixel depth so the new frame's geometry can
        //   perform correct visibility testing
        glClearColor(0.08f, 0.09f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Update and render layers in order. Because layers receive the same
        // camera and dt, the viewer acts as the orchestration point for the
        // whole scene.
        for (const auto& layer : _layers) {
            if (layer) {
                layer->update(_win, _cam, _dt);
            }
        }

        // Present the rendered back buffer to the screen. GLFW uses double
        // buffering, so drawing occurs off-screen until this swap.
        glfwSwapBuffers(_win);
    }
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
Viewer<T>::shutdown_layers() {
    // Give every layer a chance to release its own GPU or CPU resources before
    // the viewer tears down the underlying GL context.
    for (const auto& layer : _layers) {
        if (layer) {
            layer->shutdown();
        }
    }
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
Viewer<T>::cleanup_gl() {
    // Destroy the platform window first. This also releases the associated
    // OpenGL context.
    if (_win) {
        glfwDestroyWindow(_win);
        _win = nullptr;
    }

    // Terminate GLFW's global state. GLFW allows repeated terminate calls after
    // partial initialization patterns, so this is used as a broad cleanup step.
    glfwTerminate();
}

/* ====================================================================== */
/* Viewer<T>::Builder                                                      */
/* ====================================================================== */

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE typename Viewer<T>::Builder&
Viewer<T>::Builder::with_dt(T dt) noexcept {
    // dt is stored verbatim here. Semantic validation is deferred to validate()
    // so the builder remains cheap and fluent during configuration.
    _dt = dt;
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE typename Viewer<T>::Builder&
Viewer<T>::Builder::with_title(const char* title) noexcept {
    // Normalize null to the default title immediately so downstream code can
    // rely on _title being meaningful in the common path.
    _title = title ? title : "Atlas Viewer";
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE typename Viewer<T>::Builder&
Viewer<T>::Builder::with_size(int width, int height) noexcept {
    // Zero means "let the viewer decide later from monitor resolution".
    // Negative values are rejected by validate().
    _width  = width;
    _height = height;
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE typename Viewer<T>::Builder&
Viewer<T>::Builder::with_fullscreen(bool fullscreen) noexcept {
    _fullscreen = fullscreen;
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
Viewer<T>::Builder::validate() const {
    // Enforce basic runtime invariants before any windowing or graphics code is
    // touched. Failing here gives clearer diagnostics than deferring invalid
    // configuration into GLFW creation calls.
    if (_dt <= T(0)) {
        atlas::logger::error()
            << "Viewer::Builder validation failed: dt must be positive.";
        throw std::runtime_error("Viewer::Builder: dt must be positive.");
    }

    if ((_width < 0) || (_height < 0)) {
        atlas::logger::error()
            << "Viewer::Builder validation failed: width/height must be >= 0.";
        throw std::runtime_error("Viewer::Builder: width/height must be >= 0.");
    }

    if (_title == nullptr) {
        atlas::logger::error()
            << "Viewer::Builder validation failed: title must not be null.";
        throw std::runtime_error("Viewer::Builder: title must not be null.");
    }
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE Viewer<T>
Viewer<T>::Builder::build() const {
    // Build by value after validation. The resulting Viewer remains lightweight
    // because no actual GL resources are created until run() is called.
    validate();
    return Viewer<T>(_dt, _width, _height, _title, _fullscreen);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE std::shared_ptr<Viewer<T>>
Viewer<T>::Builder::make_shared() const {
    // Reuse build() so validation and default normalization remain centralized.
    auto viewer = build();
    return std::make_shared<Viewer<T>>(std::move(viewer));
}

} // namespace atlas::vizkit

#endif
