#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {

inline void
Camera::set_axis_view(const int index) noexcept {
    constexpr float pi = static_cast<float>(M_PI);
    constexpr float half_pi = pi * 0.5f;
    constexpr float top_pitch = half_pi - 0.001f;

    switch (index) {
    case 0: // front
        yaw = half_pi;
        pitch = 0.0f;
        break;
    case 1: // back
        yaw = -half_pi;
        pitch = 0.0f;
        break;
    case 2: // top
        yaw = half_pi;
        pitch = top_pitch;
        break;
    case 3: // bottom
        yaw = half_pi;
        pitch = -top_pitch;
        break;
    case 4: // left
        yaw = 0.0f;
        pitch = 0.0f;
        break;
    case 5: // right
        yaw = pi;
        pitch = 0.0f;
        break;
    case 6: // perspective
        yaw = -0.9f;
        pitch = 0.45f;
        break;
    default:
        break;
    }
}

inline void
Camera::handle_view_shortcuts(GLFWwindow* w) noexcept {
    if (w == nullptr) return;

    const int keys[7] = {
        GLFW_KEY_1,
        GLFW_KEY_2,
        GLFW_KEY_3,
        GLFW_KEY_4,
        GLFW_KEY_5,
        GLFW_KEY_6,
        GLFW_KEY_7,
    };

    for (int i = 0; i < 7; ++i) {
        const bool pressed = glfwGetKey(w, keys[i]) == GLFW_PRESS;
        if (pressed && !view_key_down[i]) {
            set_axis_view(i);
        }
        view_key_down[i] = pressed;
    }
}

inline void
Camera::init_mouse_controls(GLFWwindow* w) {
    // Initialize mouse interaction bindings for this camera.
    //
    // Responsibilities:
    // - bind this Camera instance to the GLFW window user pointer
    // - register scroll callback for zoom input
    // - initialize last known cursor position
    //
    // Early exit if:
    // - already initialized (avoid duplicate callback registration)
    // - window pointer is null
    if (mouse_initialized || w == nullptr) return;

    // Store `this` so static callbacks can retrieve the Camera instance.
    glfwSetWindowUserPointer(w, this);

    // Register scroll callback (used for zoom control).
    glfwSetScrollCallback(w, &Camera::scroll_callback);

    // Initialize cursor position to avoid large initial deltas.
    glfwGetCursorPos(w, &last_cursor_x, &last_cursor_y);

    // Mark initialization complete.
    mouse_initialized = true;
}

inline void
Camera::scroll_callback(GLFWwindow* w, double xoffset, double yoffset) {
    // Handle mouse scroll input for zooming.
    //
    // Notes:
    // - xoffset is unused (horizontal scroll ignored)
    // - yoffset controls zoom direction and magnitude
    (void)xoffset;

    // Safety check: window must be valid.
    if (w == nullptr) return;

    // Retrieve Camera instance associated with this window.
    auto* camera = static_cast<Camera*>(glfwGetWindowUserPointer(w));
    if (camera == nullptr) return;

    // Accumulate zoom input into a pending value.
    //
    // This decouples input sampling from camera update,
    // allowing smoother integration in handle().
    camera->pending_scroll_zoom -= static_cast<float>(yoffset) * camera->scroll_zoom_sensitivity;
}

inline void
Camera::handle(GLFWwindow* w) {
    // Process all user input and update camera parameters.
    //
    // Responsibilities:
    // - handle mouse drag (orbit + zoom)
    // - handle keyboard input
    // - apply scroll zoom
    // - clamp camera parameters

    // Ensure mouse system is initialized.
    init_mouse_controls(w);

    // Retrieve current cursor position.
    double cursor_x = last_cursor_x;
    double cursor_y = last_cursor_y;
    glfwGetCursorPos(w, &cursor_x, &cursor_y);

    // Compute cursor movement delta since last frame.
    const auto delta_x = static_cast<float>(cursor_x - last_cursor_x);
    const auto delta_y = static_cast<float>(cursor_y - last_cursor_y);

    // --- Left mouse button: orbit camera (yaw/pitch) ---
    const bool left_pressed = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

    if (left_pressed) {
        // Only apply rotation when dragging continuously.
        if (left_drag_active) {
            yaw -= delta_x * orbit_sensitivity;   // horizontal rotation
            pitch -= delta_y * orbit_sensitivity; // vertical rotation
        }
        left_drag_active = true;
    } else {
        // Reset drag state when button released.
        left_drag_active = false;
    }

    // --- Right mouse button: zoom via vertical drag ---
    const bool right_pressed = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    if (right_pressed) {
        if (right_drag_active) {
            dist *= expf(delta_y * drag_zoom_sensitivity); // zoom in/out
        }
        right_drag_active = true;
    } else {
        right_drag_active = false;
    }

    // Update stored cursor position for next frame.
    last_cursor_x = cursor_x;
    last_cursor_y = cursor_y;

    // --- Keyboard controls ---
    handle_view_shortcuts(w);

    // Horizontal orbit
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) yaw -= 0.02f;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) yaw += 0.02f;

    // Vertical orbit
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) pitch += 0.02f;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) pitch -= 0.02f;

    // Zoom (distance)
    if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) dist *= 1.01f;
    if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) dist *= 0.99f;

    // Apply accumulated scroll-based zoom.
    dist *= expf(pending_scroll_zoom);

    // Reset scroll accumulator after applying.
    pending_scroll_zoom = 0.0f;

    // Clamp pitch to avoid flipping (gimbal lock-like behavior).
    constexpr float max_pitch = static_cast<float>(M_PI) * 0.5f - 0.001f;
    pitch = fminf(fmaxf(pitch, -max_pitch), max_pitch);

    // Clamp distance to maintain reasonable zoom bounds.
    dist = fminf(fmaxf(dist, min_dist), max_dist);
}

inline void
Camera::build_mvp(int w, int h, float out_mvp[16]) const {
    // Build Model-View-Projection (MVP) matrix.
    //
    // Steps:
    // 1. build projection matrix (perspective)
    // 2. compute camera position from spherical coordinates
    // 3. build view matrix (look-at)
    // 4. multiply projection * view into out_mvp

    {
        // --- Projection matrix ---
        // Compute aspect ratio (avoid division by zero).
        const float aspect = (h > 0) ? static_cast<float>(w) / static_cast<float>(h) : 1.0f;

        // Field of view in radians.
        const float fov = 45.0f * static_cast<float>(M_PI) / 180.0f;

        // Perspective projection scale factor.
        const float f = 1.0f / tanf(fov * 0.5f);

        const float scene_scale = fmaxf(dist, 1.0e-6f);
        const float z_near = fmaxf(scene_scale * 1.0e-3f, 1.0e-9f);
        const float z_far = fmaxf(scene_scale * 50.0f, z_near * 10.0f);
        const float inv_depth = 1.0f / (z_near - z_far);

        // Column-major projection matrix.
        const float proj[16] = {
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (z_far + z_near) * inv_depth, -1,
            0, 0, (2.0f * z_far * z_near) * inv_depth, 0
        };

        // --- Camera position (orbit model) ---
        // Convert spherical coordinates (yaw, pitch, dist) into Cartesian.
        const float ex = dist * cosf(pitch) * cosf(yaw);
        const float ey = dist * cosf(pitch) * sinf(yaw);
        const float ez = dist * sinf(pitch);

        // Eye position (camera position).
        const Vector3F eye { target.x - ex, target.y - ey, target.z + ez };

        // Target point the camera looks at.
        const Vector3F center = target;

        // Up direction.
        const Vector3F up { 0, 0, 1 };

        // --- View matrix (look-at construction) ---
        const Vector3F fwd = normalize(center - eye);   // forward direction
        const Vector3F s   = normalize(cross(fwd, up)); // right vector
        const Vector3F u   = cross(s, fwd);             // corrected up vector

        // Column-major view matrix.
        const float view[16] = { s.x, u.x, -fwd.x, 0, s.y, u.y, -fwd.y, 0, s.z, u.z, -fwd.z, 0, -dot(s, eye), -dot(u, eye), dot(fwd, eye), 1 };

        // --- MVP = Projection * View ---
        // Matrix multiplication (column-major).
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r) {

                // Initialize output element.
                out_mvp[c * 4 + r] = 0;

                // Accumulate dot product of row (proj) and column (view).
                for (int k = 0; k < 4; ++k)
                    out_mvp[c * 4 + r] += proj[k * 4 + r] * view[c * 4 + k];
            }
    }
}

inline void
Camera::fit_bounds(const Vector3F& lower, const Vector3F& upper) noexcept {
    target = (lower + upper) * 0.5f;

    const Vector3F extent = upper - lower;
    const float diagonal = extent.length();
    const float radius = fmaxf(diagonal * 0.5f, 1.0e-6f);

    dist = radius * 2.4f;
    min_dist = radius * 0.05f;
    max_dist = radius * 40.0f;

    const float normalized_scene_scale = fminf(fmaxf(radius, 1.0e-6f), 1.0f);
    orbit_sensitivity = 0.0008f + 0.0012f * normalized_scene_scale;
    drag_zoom_sensitivity = 0.0012f;
    scroll_zoom_sensitivity = 0.06f;

    if (!(min_dist > 0.0f)) min_dist = 1.0e-6f;
    if (!(max_dist > min_dist)) max_dist = min_dist * 100.0f;

    yaw = -0.9f;
    pitch = 0.45f;
}

} // namespace atlas::vizkit

#endif
