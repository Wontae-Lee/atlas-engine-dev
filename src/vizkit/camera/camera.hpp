#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {
void
Camera::handle(GLFWwindow* w) {

    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) yaw -= 0.02f;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) yaw += 0.02f;
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) pitch += 0.02f;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) pitch -= 0.02f;
    if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) dist += 0.05f;
    if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) dist -= 0.05f;
    pitch = fminf(fmaxf(pitch, -1.2f), 1.2f);
    dist  = fminf(fmaxf(dist, 2.0f), 100.0f);
}

void
Camera::build_mvp(int w, int h, float out_mvp[16]) const {
    {
        float aspect   = (h > 0) ? static_cast<float>(w) / static_cast<float>(h) : 1.0f;
        float fov      = 45.0f * static_cast<float>(M_PI) / 180.0f;
        float f        = 1.0f / tanf(fov * 0.5f);
        float proj[16] = { f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, -1, -1, 0, 0, -0.2f, 0 };
        float ex       = dist * cosf(pitch) * cosf(yaw);
        float ey       = dist * cosf(pitch) * sinf(yaw);
        float ez       = dist * sinf(pitch);
        Vector3F eye { -ex, -ey, ez }, center { 0, 0, 0.6f }, up { 0, 0, 1 };
        Vector3F fwd   = normalize(center - eye);
        Vector3F s     = normalize(cross(fwd, up));
        Vector3F u     = cross(s, fwd);
        float view[16] = {
            s.x,
            u.x,
            -fwd.x,
            0,
            s.y,
            u.y,
            -fwd.y,
            0,
            s.z,
            u.z,
            -fwd.z,
            0,
            -dot(s, eye),
            -dot(u, eye),
            dot(fwd, eye),
            1
        };
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
