#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {
void
Camera::init_mouse_controls(GLFWwindow* w) {
    if (mouse_initialized || w == nullptr) return;

    glfwSetWindowUserPointer(w, this);
    glfwSetScrollCallback(w, &Camera::scroll_callback);
    glfwGetCursorPos(w, &last_cursor_x, &last_cursor_y);
    mouse_initialized = true;
}

void
Camera::scroll_callback(GLFWwindow* w, double xoffset, double yoffset) {
    (void)xoffset;

    if (w == nullptr) return;

    auto* camera = static_cast<Camera*>(glfwGetWindowUserPointer(w));
    if (camera == nullptr) return;

    camera->pending_scroll_zoom -= static_cast<float>(yoffset) * 0.8f;
}

void
Camera::handle(GLFWwindow* w) {
    init_mouse_controls(w);

    double cursor_x = last_cursor_x;
    double cursor_y = last_cursor_y;
    glfwGetCursorPos(w, &cursor_x, &cursor_y);

    const float delta_x = static_cast<float>(cursor_x - last_cursor_x);
    const float delta_y = static_cast<float>(cursor_y - last_cursor_y);

    const bool left_pressed = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (left_pressed) {
        if (left_drag_active) {
            yaw -= delta_x * 0.008f;
            pitch -= delta_y * 0.008f;
        }
        left_drag_active = true;
    } else {
        left_drag_active = false;
    }

    const bool right_pressed = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (right_pressed) {
        if (right_drag_active) {
            dist += delta_y * 0.05f;
        }
        right_drag_active = true;
    } else {
        right_drag_active = false;
    }

    last_cursor_x = cursor_x;
    last_cursor_y = cursor_y;

    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) yaw -= 0.02f;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) yaw += 0.02f;

    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) pitch += 0.02f;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) pitch -= 0.02f;

    if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) dist += 0.05f;
    if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) dist -= 0.05f;
    dist += pending_scroll_zoom;
    pending_scroll_zoom = 0.0f;

    pitch = fminf(fmaxf(pitch, -1.2f), 1.2f);

    dist = fminf(fmaxf(dist, 2.0f), 100.0f);
}

void
Camera::build_mvp(int w, int h, float out_mvp[16]) const {
    {

        const float aspect = (h > 0) ? static_cast<float>(w) / static_cast<float>(h) : 1.0f;

        const float fov = 45.0f * static_cast<float>(M_PI) / 180.0f;

        const float f = 1.0f / tanf(fov * 0.5f);

        const float proj[16] = { f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, -1, -1, 0, 0, -0.2f, 0 };

        const float ex = dist * cosf(pitch) * cosf(yaw);
        const float ey = dist * cosf(pitch) * sinf(yaw);
        const float ez = dist * sinf(pitch);

        const Vector3F eye { -ex, -ey, ez }, center { 0, 0, 0.6f }, up { 0, 0, 1 };

        const Vector3F fwd = normalize(center - eye);
        const Vector3F s   = normalize(cross(fwd, up));
        const Vector3F u   = cross(s, fwd);

        const float view[16] = { s.x, u.x, -fwd.x, 0, s.y, u.y, -fwd.y, 0, s.z, u.z, -fwd.z, 0, -dot(s, eye), -dot(u, eye), dot(fwd, eye), 1 };

        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r) {

                out_mvp[c * 4 + r] = 0;

                for (int k = 0; k < 4; ++k)
                    out_mvp[c * 4 + r] += proj[k * 4 + r] * view[c * 4 + k];
            }
    }
}
}

#endif