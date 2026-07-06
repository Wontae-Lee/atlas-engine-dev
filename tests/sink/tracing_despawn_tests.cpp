#include <atlas/sink/tracing_despawn.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>

namespace {

using atlas::Box;
using atlas::Geometry;
using atlas::TracingDespawn;
using atlas::Vector3;

Geometry
make_box_operator() {
    static const auto box = Box::builder()
                                .with_lower_corner(Vector3(2.0f, -1.0f, -1.0f))
                                .with_upper_corner(Vector3(3.0f, 1.0f, 1.0f))
                                .build();
    return Geometry(box);
}

}

TEST(TracingDespawn, DetectsVelocityTraceIntersectionsWithinTime) {
    const auto geometry_operator = make_box_operator();

    EXPECT_TRUE(TracingDespawn::despawn(
        geometry_operator, Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 0.0f, 0.0f), 2.5f));
    EXPECT_FALSE(TracingDespawn::despawn(
        geometry_operator, Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 0.0f, 0.0f), 1.5f));
    EXPECT_FALSE(TracingDespawn::despawn(
        geometry_operator, Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 1.0f, 0.0f), 5.0f));
    EXPECT_TRUE(TracingDespawn::despawn(
        geometry_operator, Vector3(1.0f, 0.0f, 0.0f), Vector3(1.0f, 0.0f, 0.0f), 1.5f));
    EXPECT_FALSE(TracingDespawn::despawn(
        geometry_operator, Vector3(-2.0f, 0.0f, 0.0f), Vector3(1.0f, 0.0f, 0.0f), 2.5f));
}
