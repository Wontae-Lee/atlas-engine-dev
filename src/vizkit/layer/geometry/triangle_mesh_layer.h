#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <vizkit/layer/geometry/geometry_layer.h>

namespace atlas::vizkit {

template <typename T>
class TriangleMeshLayer final : public GeometryLayer<T> {

public:
    class Builder;

    TriangleMeshLayer(const atlas::UnitHostPtr<T>& unit);

    static Builder
    builder();

protected:
    void
    build_geometry(std::vector<Vector3<T>>& pos) override;
};

template <typename T>
class TriangleMeshLayer<T>::Builder {

public:
    Builder&
    with_unit(const atlas::UnitHostPtr<T>& u);

    TriangleMeshLayer
    build() const;

    std::shared_ptr<TriangleMeshLayer>
    make_shared() const;

private:
    void
    validate() const;

    atlas::UnitHostPtr<T> _unit = nullptr;
};

}

#include <vizkit/layer/geometry/triangle_mesh_layer.hpp>

#endif