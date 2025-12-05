#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
#include <atlas/vizkit/camera/camera.h>
#include <atlas/vizkit/layer/layer.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

template <typename T>
class Viewer {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE
    Viewer(int width = 1280, int height = 720, const char* title = "Atlas Viewer");

    ATLAS_HOST ATLAS_FORCE_INLINE ~Viewer();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_layer(const std::shared_ptr<Layer<T>>& layer);

    ATLAS_HOST ATLAS_FORCE_INLINE int
    run();

private:
    int _width;
    int _height;
    const char* _title;

    GLFWwindow* _win { nullptr };
    Camera _cam;
    std::vector<std::shared_ptr<Layer<T>>> _layers;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_gl();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_layers();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    main_loop();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    shutdown_layers();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    cleanup_gl();
};

}

#include <atlas/vizkit/viewer/viewer.hpp>

#endif
