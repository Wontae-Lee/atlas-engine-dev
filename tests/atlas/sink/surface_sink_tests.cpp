#include <atlas/sink/surface_sink.h>

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
using atlas::SurfaceSink;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::Unit;
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

TEST(SurfaceSink, BuilderConfiguresState) {
    const auto sink = SurfaceSink::builder()
                          .with_unit(make_static_box_unit())
                          .with_tolerance(0.01f)
                          .build();

    EXPECT_FALSE(sink.unit().dynamic());
}

TEST(SurfaceSink, BuilderRejectsMissingUnit) {
    EXPECT_THROW(
        static_cast<void>(SurfaceSink::builder().with_tolerance(0.01f).build()),
        std::runtime_error);
}

TEST(SurfaceSink, BuilderRejectsNegativeTolerance) {
    EXPECT_THROW(
        static_cast<void>(SurfaceSink::builder().with_unit(make_static_box_unit()).with_tolerance(-1.0f).build()),
        std::runtime_error);
}

TEST(SurfaceSink, MakeHostSharedBuildsSink) {
    const auto sink = SurfaceSink::builder()
                          .with_unit(make_static_box_unit())
                          .make_host_shared();

    ASSERT_TRUE(static_cast<bool>(sink));
    EXPECT_FALSE(sink->unit().dynamic());
}

TEST(SurfaceSink, DespawnDetectsParticleOnSurface) {
    const auto sink = SurfaceSink::builder()
                          .with_unit(make_static_box_unit())
                          .with_tolerance(0.01f)
                          .build();

    // On the +x face of the box.
    EXPECT_TRUE(sink.despawn(Float3(1.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));
    // Deep inside the box, not on the surface.
    EXPECT_FALSE(sink.despawn(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));
}

TEST(SurfaceSink, DespawnMissesParticleOffSurface) {
    const auto sink = SurfaceSink::builder()
                          .with_unit(make_static_box_unit())
                          .with_tolerance(0.5f)
                          .build();

    // 1.0 beyond the +x face, well past the tolerance band.
    EXPECT_FALSE(sink.despawn(Float3(2.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));
}

TEST(SurfaceSink, DespawnAcceptsParticleAtToleranceBoundary) {
    const auto sink = SurfaceSink::builder()
                          .with_unit(make_static_box_unit())
                          .with_tolerance(0.5f)
                          .build();

    // Exactly tolerance (0.5) beyond the +x face; the band is inclusive.
    EXPECT_TRUE(sink.despawn(Float3(1.5f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));
}

TEST(SurfaceSink, AdvanceMovesUnit) {
    auto sink = SurfaceSink::builder()
                    .with_unit(make_dynamic_box_unit(Float3(0.0f, 0.0f, 1.0f)))
                    .build();

    sink.advance(0.5f);

    EXPECT_NEAR(sink.unit().sync().translation.z, 0.5f, tol);
}

TEST(SurfaceSink, DespawnFollowsMovedUnit) {
    // The surface test runs in the unit's local frame, so advancing the boundary
    // must move where the despawn band lies in world space.
    auto sink = SurfaceSink::builder()
                    .with_unit(make_dynamic_box_unit(Float3(1.0f, 0.0f, 0.0f)))
                    .with_tolerance(0.01f)
                    .build();

    // Before moving, the +x face sits at world x = 1.
    EXPECT_TRUE(sink.despawn(Float3(1.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));

    sink.advance(1.0f); // translation.x -> 1, so the +x face is now at world x = 2.

    EXPECT_TRUE(sink.despawn(Float3(2.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));
    // The old surface point is now the box centre in local space: no longer on the surface.
    EXPECT_FALSE(sink.despawn(Float3(1.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));
}
