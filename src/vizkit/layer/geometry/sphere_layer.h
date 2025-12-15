#pragma once

#include <atlas/core/macros.h>
#include <vizkit/layer/geometry/geometry_layer.h>
#include <atlas/math/math.h>

#include <vector>

namespace atlas::vizkit {


template <typename T>
class SphereLayer final : public GeometryLayer<T> {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE
    SphereLayer(const Vector3<T>& center,
                T radius,
                int slices = 32,
                int stacks = 16);

    ATLAS_HOST ATLAS_FORCE_INLINE
    ~SphereLayer() override = default;

protected:
    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    Vector3<T> _center;
    T _radius;
    int _slices;
    int _stacks;
};

}

#include <vizkit/layer/geometry/sphere_layer.hpp>