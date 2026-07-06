#include <atlas/sink/surface_despawn.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>

namespace {

using atlas::Box;
using atlas::Geometry;
using atlas::SurfaceDespawn;
using atlas::Vector3;

Geometry
make_box_operator() {
    static const auto box = Box::builder()
                                .with_lower_corner(Vector3(-1.0f, -1.0f, -1.0f))
                                .with_upper_corner(Vector3(1.0f, 1.0f, 1.0f))
                                .build();
    return Geometry(box);
}

}

TEST(SurfaceDespawn, DetectsSurfacePoints) {
    const auto geometry_operator = make_box_operator();

    EXPECT_TRUE(SurfaceDespawn::despawn(geometry_operator, Vector3(1.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_FALSE(SurfaceDespawn::despawn(geometry_operator, Vector3(0.0f, 0.0f, 0.0f), 0.0f));
}
