#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {

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
    camera->pending_scroll_zoom -= static_cast<float>(yoffset) * 0.8f;
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
            yaw -= delta_x * 0.008f;   // horizontal rotation
            pitch -= delta_y * 0.008f; // vertical rotation
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
            dist += delta_y * 0.05f; // zoom in/out
        }
        right_drag_active = true;
    } else {
        right_drag_active = false;
    }

    // Update stored cursor position for next frame.
    last_cursor_x = cursor_x;
    last_cursor_y = cursor_y;

    // --- Keyboard controls ---
    // Horizontal orbit
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) yaw -= 0.02f;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) yaw += 0.02f;

    // Vertical orbit
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) pitch += 0.02f;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) pitch -= 0.02f;

    // Zoom (distance)
    if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) dist += 0.05f;
    if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) dist -= 0.05f;

    // Apply accumulated scroll-based zoom.
    dist += pending_scroll_zoom;

    // Reset scroll accumulator after applying.
    pending_scroll_zoom = 0.0f;

    // Clamp pitch to avoid flipping (gimbal lock-like behavior).
    pitch = fminf(fmaxf(pitch, -1.2f), 1.2f);

    // Clamp distance to maintain reasonable zoom bounds.
    dist = fminf(fmaxf(dist, 2.0f), 100.0f);
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

        // Column-major projection matrix.
        const float proj[16] = { f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, -1, -1, 0, 0, -0.2f, 0 };

        // --- Camera position (orbit model) ---
        // Convert spherical coordinates (yaw, pitch, dist) into Cartesian.
        const float ex = dist * cosf(pitch) * cosf(yaw);
        const float ey = dist * cosf(pitch) * sinf(yaw);
        const float ez = dist * sinf(pitch);

        // Eye position (camera position).
        const Vector3F eye { -ex, -ey, ez };

        // Target point the camera looks at.
        const Vector3F center { 0, 0, 0.6f };

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

} // namespace atlas::vizkit

#endif