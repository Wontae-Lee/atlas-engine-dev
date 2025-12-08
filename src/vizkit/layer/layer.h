#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/macros/macros.h>
#include <vizkit/camera/camera.h>
namespace atlas::vizkit {

template <typename T>
class Layer {
public:
    virtual ~Layer() = default;

    virtual void
    init(GLFWwindow* window, Camera& camera)
        = 0;

    virtual void
    update(GLFWwindow* window, Camera& camera)
        = 0;

    virtual void
    shutdown() { }
};

}

#endif
