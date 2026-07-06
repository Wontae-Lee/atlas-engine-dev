#include <atlas/sink/volume_despawn.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>

namespace {

using atlas::Box;
using atlas::Geometry;
using atlas::Float3;
using atlas::VolumeDespawn;

Geometry
make_box_operator() {
    static const auto box = Box::builder()
                                .with_lower_corner(Float3(-1.0f, -1.0f, -1.0f))
                                .with_upper_corner(Float3(1.0f, 1.0f, 1.0f))
                                .build();
    return Geometry(box);
}

}

TEST(VolumeDespawn, DetectsInteriorPoints) {
    const auto geometry_operator = make_box_operator();

    EXPECT_TRUE(VolumeDespawn::despawn(geometry_operator, Float3(0.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_FALSE(VolumeDespawn::despawn(geometry_operator, Float3(3.0f, 0.0f, 0.0f), 0.0f));
}
