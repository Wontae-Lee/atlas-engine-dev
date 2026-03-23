#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <vizkit/layer/geometry/geometry_layer.h>

namespace atlas::vizkit {

template <typename T>
class TriangleLayer final : public GeometryLayer<T> {
public:
    class Builder;

    TriangleLayer(const atlas::UnitHostPtr<T>& unit);

    static Builder
    builder();

protected:
    void
    build_geometry(std::vector<Vector3<T>>& pos) override;
};

template <typename T>
class TriangleLayer<T>::Builder {
public:
    Builder&
    with_unit(const atlas::UnitHostPtr<T>& u);

    TriangleLayer
    build() const;

    std::shared_ptr<TriangleLayer>
    make_shared() const;

private:
    void
    validate() const;

    atlas::UnitHostPtr<T> _unit = nullptr;
};

}

#include <vizkit/layer/geometry/triangle_layer.hpp>

#endif