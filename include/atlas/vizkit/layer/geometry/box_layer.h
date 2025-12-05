
#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/vizkit/layer/geometry/geometry_layer.h>

namespace atlas::vizkit {

template <typename T>
class BoxLayer final : public GeometryLayer<T> {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE
    BoxLayer(const Vector3<T>& min_corner, const Vector3<T>& max_corner);

    ATLAS_HOST ATLAS_FORCE_INLINE ~BoxLayer() override = default;

protected:
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    Vector3<T> _min_corner;
    Vector3<T> _max_corner;
};

}

#include <atlas/vizkit/layer/geometry/box_layer.hpp>
