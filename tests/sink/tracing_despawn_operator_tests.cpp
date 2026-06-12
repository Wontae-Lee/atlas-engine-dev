#include "../utilities/test_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/sink/tracing_despawn_operator.h>

#include <testkit/testkit.h>

namespace {

using atlas::Box;
using atlas::GeometryOperator;
using atlas::Vector3F;
using atlas::TracingDespawnOperator;

GeometryOperator<float>
make_box_operator() {
    static const auto box = Box<float>::builder()
                                .with_lower_corner(Vector3F(2, -1, -1))
                                .with_upper_corner(Vector3F(3, 1, 1))
                                .build();
    return box.make_device_geometry_view();
}

} // namespace

TEST(TracingDespawnOperator, DetectsVelocityTraceIntersectionsWithinTime) {
    const auto geometry_operator = make_box_operator();

    EXPECT_TRUE(TracingDespawnOperator<float>::despawn(
        geometry_operator, Vector3F(0, 0, 0), Vector3F(1, 0, 0), 2.5f));
    EXPECT_FALSE(TracingDespawnOperator<float>::despawn(
        geometry_operator, Vector3F(0, 0, 0), Vector3F(1, 0, 0), 1.5f));
    EXPECT_FALSE(TracingDespawnOperator<float>::despawn(
        geometry_operator, Vector3F(0, 0, 0), Vector3F(0, 1, 0), 5.0f));
    EXPECT_TRUE(TracingDespawnOperator<float>::despawn(
        geometry_operator, Vector3F(1, 0, 0), Vector3F(1, 0, 0), 1.5f));
    EXPECT_FALSE(TracingDespawnOperator<float>::despawn(
        geometry_operator, Vector3F(-2, 0, 0), Vector3F(1, 0, 0), 2.5f));
}
