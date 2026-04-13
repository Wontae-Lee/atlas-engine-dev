#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <vizkit/macros/macros.h>

namespace atlas::vizkit {

struct Camera {

    float yaw = 0.0f, pitch = 0.5f, dist = 20.0f;

    bool mouse_initialized = false;

    bool left_drag_active = false;

    bool right_drag_active = false;

    double last_cursor_x = 0.0;

    double last_cursor_y = 0.0;

    float pending_scroll_zoom = 0.0f;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    handle(GLFWwindow* w);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_mvp(int w, int h, float out_mvp[16]) const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_mouse_controls(GLFWwindow* w);

    ATLAS_HOST static void
    scroll_callback(GLFWwindow* w, double xoffset, double yoffset);
};

}

#include <vizkit/camera/camera.hpp>

#endif