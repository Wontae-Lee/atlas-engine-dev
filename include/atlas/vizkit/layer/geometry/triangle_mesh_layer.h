#ifndef ATLAS_ENGINE_DEV_TRIANGLE_MESH_LAYER_H
#define ATLAS_ENGINE_DEV_TRIANGLE_MESH_LAYER_H

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/vizkit/layer/geometry/geometry_layer.h>

#include <vector>

namespace atlas::vizkit {


template <typename T>
class TriangleMeshLayer final : public GeometryLayer<T> {
public:

    ATLAS_HOST ATLAS_FORCE_INLINE explicit TriangleMeshLayer(std::vector<Vector3<T>> vertices);

    ATLAS_HOST ATLAS_FORCE_INLINE ~TriangleMeshLayer() override = default;

protected:

    void
    build_geometry(std::vector<Vector3<T>>& positions) override;

private:
    std::vector<Vector3<T>> _vertices;
};

}

#include <atlas/vizkit/layer/geometry/triangle_mesh_layer.hpp>

#endif