#include <atlas/sink/volume_sink.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <utility>

namespace {

using atlas::Box;
using atlas::Geometry;
using atlas::Quaternion;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::Unit;
using atlas::VolumeSink;
using atlas::Float3;
using atlas::tol;

SyncHostPtr
make_identity_sync() {
    return Sync::builder()
        .with_rigid_pose(Float3(0.0f, 0.0f, 0.0f), Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
        .make_host_shared();
}

Geometry
make_unit_box() {
    return Geometry(Box::builder()
                        .with_lower_corner(Float3(-1.0f, -1.0f, -1.0f))
                        .with_upper_corner(Float3(1.0f, 1.0f, 1.0f))
                        .build());
}

Unit
make_static_box_unit() {
    return Unit::builder()
        .with_geometry(make_unit_box())
        .with_sync(make_identity_sync())
        .build();
}

Unit
make_dynamic_box_unit(const Float3& velocity) {
    return Unit::builder()
        .with_geometry(make_unit_box())
        .with_sync(make_identity_sync())
        .with_velocity(velocity)
        .build();
}

}

TEST(VolumeSink, BuilderConfiguresState) {
    const auto sink = VolumeSink::builder()
                          .with_unit(make_static_box_unit())
                          .build();

    EXPECT_FALSE(sink.unit().dynamic());
}

TEST(VolumeSink, BuilderRejectsMissingUnit) {
    EXPECT_THROW(
        static_cast<void>(VolumeSink::builder().build()),
        std::runtime_error);
}

TEST(VolumeSink, BuilderRejectsNegativeTolerance) {
    EXPECT_THROW(
        static_cast<void>(VolumeSink::builder().with_unit(make_static_box_unit()).with_tolerance(-1.0f).build()),
        std::runtime_error);
}

TEST(VolumeSink, DespawnDetectsParticleInsideVolume) {
    const auto sink = VolumeSink::builder()
                          .with_unit(make_static_box_unit())
                          .build();

    EXPECT_TRUE(sink.despawn(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_FALSE(sink.despawn(Float3(5.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));
}

TEST(VolumeSink, AdvanceMovesUnit) {
    auto sink = VolumeSink::builder()
                    .with_unit(make_dynamic_box_unit(Float3(0.0f, 0.0f, 1.0f)))
                    .build();

    sink.advance(0.5f);

    EXPECT_NEAR(sink.unit().sync().translation.z, 0.5f, tol);
}
