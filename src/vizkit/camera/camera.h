#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
#include <atlas/math/math.h>
#include <vizkit/macros/macros.h>

namespace atlas::vizkit {

/**
 * @brief Simple orbit camera used by VizKit viewers.
 *
 * The camera orbits around a fixed scene target near the world origin and
 * exposes a minimal interactive control surface:
 * - `yaw` rotates the camera around the world Z axis
 * - `pitch` tilts the camera upward or downward
 * - `dist` controls the distance from the orbit target
 *
 * The view is built from these spherical-style parameters and always looks
 * toward a fixed center point in the scene. This makes the type suitable for
 * lightweight visualization tools where a compact, dependency-free camera
 * state is preferred over a full scene graph camera abstraction.
 */
struct Camera {
    /** @brief Azimuth angle in radians used for horizontal orbiting. */
    float yaw = 0.0f, pitch = 0.5f, dist = 20.0f;

    /**
     * @brief Updates the camera state from keyboard input.
     *
     * @param w Active GLFW window used to query key states.
     *
     * The function polls the keyboard and applies small incremental changes:
     * - `A`: decrease yaw
     * - `D`: increase yaw
     * - `W`: increase pitch
     * - `S`: decrease pitch
     * - `Q`: increase distance
     * - `E`: decrease distance
     *
     * After applying input, the camera state is clamped to stable operating
     * ranges:
     * - `pitch` is limited to `[-1.2, 1.2]` radians
     * - `dist` is limited to `[2.0, 100.0]`
     *
     * @note This function does not perform event handling; it only samples the
     * current key state from GLFW.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    handle(GLFWwindow* w);

    /**
     * @brief Builds a combined model-view-projection matrix for rendering.
     *
     * @param w Viewport width in pixels.
     * @param h Viewport height in pixels.
     * @param out_mvp Output buffer that receives the resulting 4x4 matrix.
     *
     * The generated matrix combines:
     * - a perspective projection with a 45 degree vertical field of view
     * - an orbit-style view transform derived from @ref yaw, @ref pitch, and
     *   @ref dist
     *
     * The camera looks from the computed eye position toward a fixed target
     * near the scene origin, using world +Z as the up direction.
     *
     * `out_mvp` must point to storage for 16 contiguous `float` values. The
     * matrix is written in the same column-major layout used by the internal
     * multiplication code and expected by typical OpenGL upload paths.
     *
     * @note If `h <= 0`, the aspect ratio falls back to `1.0f` to avoid a
     * division by zero during projection setup.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_mvp(int w, int h, float out_mvp[16]) const;
};

}

#include <vizkit/camera/camera.hpp>

#endif
