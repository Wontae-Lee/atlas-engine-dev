#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
#include <atlas/core/macros.h>

namespace atlas::vizkit {

struct Camera {
    float yaw = 0.0f, pitch = 0.5f, dist = 20.0f;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    handle(GLFWwindow* w);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_mvp(int w, int h, float out_mvp[16]) const;
};

}

#include <atlas/vizkit/camera/camera.hpp>

#endif
