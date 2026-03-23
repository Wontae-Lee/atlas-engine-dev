#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/camera/camera.h>
#include <vizkit/layer/layer.h>
#include <vizkit/macros/macros.h>

#include <memory>
#include <vector>

namespace atlas::vizkit {

template <typename T>
class Viewer final {
public:
    class Builder;

public:
    Viewer() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Viewer(T dt,
           int width,
           int height,
           const char* title,
           bool fullscreen) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE ~Viewer();

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_layer(const std::shared_ptr<Layer<T>>& layer);

    ATLAS_HOST ATLAS_FORCE_INLINE int
    run();

private:
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

private:
    T _dt = static_cast<T>(0.01);

    int _width = 0;

    int _height = 0;

    const char* _title = "Atlas Viewer";

    bool _fullscreen = false;

    GLFWwindow* _win = nullptr;

    Camera _cam {};

    std::vector<std::shared_ptr<Layer<T>>> _layers;
};

template <typename T>
class Viewer<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_dt(T dt) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_title(const char* title) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_size(int width, int height) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fullscreen(bool fullscreen = true) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Viewer<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE std::shared_ptr<Viewer<T>>
    make_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    T _dt = static_cast<T>(0.01);

    int _width = 0;

    int _height = 0;

    const char* _title = "Atlas Viewer";

    bool _fullscreen = false;
};

}

#include <vizkit/viewer/viewer.hpp>

#endif