#include "../utilities/test_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/sink/surface_despawn_operator.h>

#include <testkit/testkit.h>

namespace {

using atlas::Box;
using atlas::GeometryOperator;
using atlas::Vector3F;
using atlas::SurfaceDespawnOperator;

GeometryOperator<float>
make_box_operator() {
    static const auto box = Box<float>::builder()
                                .with_lower_corner(Vector3F(-1, -1, -1))
                                .with_upper_corner(Vector3F(1, 1, 1))
                                .build();
    return box.make_device_geometry_view();
}

} // namespace

TEST(SurfaceDespawnOperator, DetectsSurfacePoints) {
    const auto geometry_operator = make_box_operator();

    EXPECT_TRUE(SurfaceDespawnOperator<float>::despawn(geometry_operator, Vector3F(1, 0, 0), 0.0f));
    EXPECT_FALSE(SurfaceDespawnOperator<float>::despawn(geometry_operator, Vector3F(0, 0, 0), 0.0f));
}
