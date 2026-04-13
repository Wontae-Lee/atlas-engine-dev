#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file camera.h
 * @brief Declares a lightweight interactive camera used by Vizkit rendering layers.
 *
 * @details
 * This header defines @ref atlas::vizkit::Camera, a minimal orbit-style camera
 * controller designed for interactive visualization using GLFW-based input.
 *
 * ## Purpose
 * The camera is responsible for:
 * - maintaining a simple view configuration (yaw, pitch, distance),
 * - processing mouse and scroll input from a GLFW window,
 * - producing a model-view-projection (MVP) matrix for rendering.
 *
 * It is intentionally lightweight and self-contained, making it suitable for:
 * - debug visualization,
 * - geometry inspection,
 * - simulation playback tools,
 * - integration with @ref atlas::vizkit::Layer implementations.
 *
 * ## Camera model
 * The camera follows a typical **orbit (arcball-like) model**:
 * - the camera orbits around a focal point (usually the origin),
 * - `yaw` controls horizontal rotation,
 * - `pitch` controls vertical rotation,
 * - `dist` controls zoom distance from the target.
 *
 * ## Input handling
 * Input is handled via GLFW:
 * - left mouse drag → orbit rotation (yaw/pitch),
 * - right mouse drag → optional alternate interaction (implementation-defined),
 * - scroll wheel → zoom in/out via `dist`,
 * - cursor state is tracked internally to compute deltas.
 *
 * The camera stores internal flags to ensure proper initialization and smooth
 * interaction across frames.
 *
 * ## MVP construction
 * The camera produces a 4x4 MVP matrix suitable for OpenGL-style rendering:
 * - view matrix derived from yaw/pitch/dist,
 * - projection matrix derived from viewport dimensions,
 * - combined into a single column-major float array.
 *
 * ## Usage pattern
 * Typical usage in a render loop:
 * @code
 * camera.handle(window);
 * camera.build_mvp(width, height, mvp);
 * @endcode
 *
 * ## Notes
 * - This camera is not intended to be physically accurate or feature-complete.
 * - It prioritizes simplicity and responsiveness for visualization tasks.
 *
 * ---
 */

#include <vizkit/macros/macros.h>

namespace atlas::vizkit {

/**
 * @brief Interactive orbit-style camera for visualization.
 *
 * @details
 * Stores camera orientation and interaction state, processes GLFW input, and
 * generates MVP matrices for rendering.
 *
 * The camera orbits around an implicit focal point (typically the origin) using
 * spherical coordinates defined by yaw, pitch, and distance.
 */
struct Camera {

    /**
     * @brief Horizontal rotation angle (radians or implementation-defined units).
     *
     * @details
     * Controls rotation around the vertical axis.
     */
    float yaw = 0.0f;

    /**
     * @brief Vertical rotation angle.
     *
     * @details
     * Controls elevation of the camera. Typically clamped to avoid gimbal lock
     * or flipping at extreme angles.
     */
    float pitch = 0.5f;

    /**
     * @brief Distance from the focal point.
     *
     * @details
     * Controls zoom level. Larger values move the camera further away.
     */
    float dist = 20.0f;

    /**
     * @brief Whether mouse input has been initialized.
     *
     * @details
     * Used to lazily initialize cursor tracking when the camera first receives
     * input.
     */
    bool mouse_initialized = false;

    /**
     * @brief Whether left mouse dragging is active.
     *
     * @details
     * Typically used for orbit rotation.
     */
    bool left_drag_active = false;

    /**
     * @brief Whether right mouse dragging is active.
     *
     * @details
     * May be used for alternative interactions such as panning (implementation-defined).
     */
    bool right_drag_active = false;

    /**
     * @brief Last recorded cursor x-position.
     */
    double last_cursor_x = 0.0;

    /**
     * @brief Last recorded cursor y-position.
     */
    double last_cursor_y = 0.0;

    /**
     * @brief Accumulated scroll input for zooming.
     *
     * @details
     * Updated via the GLFW scroll callback and consumed during @ref handle.
     */
    float pending_scroll_zoom = 0.0f;

    /**
     * @brief Process user input and update camera state.
     *
     * @details
     * Reads input from the provided GLFW window and updates:
     * - yaw/pitch based on mouse drag,
     * - distance based on scroll input,
     * - internal cursor tracking state.
     *
     * This function should be called once per frame before rendering.
     *
     * @param w Pointer to the active GLFW window.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    handle(GLFWwindow* w);

    /**
     * @brief Build a model-view-projection (MVP) matrix.
     *
     * @details
     * Computes a combined MVP matrix based on:
     * - current yaw/pitch/dist camera parameters,
     * - viewport dimensions.
     *
     * The resulting matrix is written into a flat array of 16 floats,
     * typically in column-major order for OpenGL usage.
     *
     * @param w Viewport width in pixels.
     * @param h Viewport height in pixels.
     * @param out_mvp Output array of size 16 receiving the MVP matrix.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_mvp(int w, int h, float out_mvp[16]) const;

private:
    /**
     * @brief Initialize mouse interaction state.
     *
     * @details
     * Sets up cursor tracking and installs callbacks as needed. This is called
     * lazily during the first invocation of @ref handle.
     *
     * @param w Pointer to the GLFW window.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_mouse_controls(GLFWwindow* w);

    /**
     * @brief GLFW scroll callback used to capture zoom input.
     *
     * @details
     * Accumulates scroll offsets into @ref pending_scroll_zoom, which is later
     * consumed during @ref handle.
     *
     * @param w GLFW window pointer.
     * @param xoffset Horizontal scroll offset (unused).
     * @param yoffset Vertical scroll offset.
     */
    ATLAS_HOST static void
    scroll_callback(GLFWwindow* w, double xoffset, double yoffset);
};

} // namespace atlas::vizkit

#include <vizkit/camera/camera.hpp>

#endif