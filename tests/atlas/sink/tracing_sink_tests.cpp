#include <atlas/sink/tracing_sink.h>

#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/math/math.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using atlas::Geometry;
using atlas::Plane;
using atlas::Quaternion;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::TracingSink;
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
make_plane_geometry() {
    return Geometry(Plane(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f)));
}

Unit
make_static_plane_unit() {
    return Unit::builder()
        .with_geometry(make_plane_geometry())
        .with_sync(make_identity_sync())
        .build();
}

Unit
make_dynamic_plane_unit(const Float3& velocity) {
    return Unit::builder()
        .with_geometry(make_plane_geometry())
        .with_sync(make_identity_sync())
        .with_velocity(velocity)
        .build();
}

}

TEST(TracingSink, BuilderRejectsMissingUnit) {
    EXPECT_THROW(
        static_cast<void>(TracingSink::builder().build()),
        std::runtime_error);
}

TEST(TracingSink, BuilderConfiguresState) {
    const auto sink = TracingSink::builder()
                          .with_unit(make_static_plane_unit())
                          .build();

    EXPECT_FALSE(sink.unit().dynamic());
}

TEST(TracingSink, MakeHostSharedBuildsSink) {
    const auto sink = TracingSink::builder()
                          .with_unit(make_static_plane_unit())
                          .make_host_shared();

    ASSERT_TRUE(static_cast<bool>(sink));
    EXPECT_FALSE(sink->unit().dynamic());
}

TEST(TracingSink, DespawnDetectsParticleReachingSurface) {
    const auto sink = TracingSink::builder()
                          .with_unit(make_static_plane_unit())
                          .build();

    // Travels distance 1 toward the plane; sweep length (speed * dt = 2) covers it.
    EXPECT_TRUE(sink.despawn(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, -1.0f), 2.0f));
}

TEST(TracingSink, DespawnMissesRecedingParticle) {
    const auto sink = TracingSink::builder()
                          .with_unit(make_static_plane_unit())
                          .build();

    EXPECT_FALSE(sink.despawn(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, 1.0f), 2.0f));
}

TEST(TracingSink, DespawnMissesWhenOutOfReach) {
    const auto sink = TracingSink::builder()
                          .with_unit(make_static_plane_unit())
                          .build();

    // Sweep length (speed * dt = 0.5) is shorter than the distance to the plane (1).
    EXPECT_FALSE(sink.despawn(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, -1.0f), 0.5f));
}

TEST(TracingSink, DespawnMissesWithZeroTimeStep) {
    const auto sink = TracingSink::builder()
                          .with_unit(make_static_plane_unit())
                          .build();

    EXPECT_FALSE(sink.despawn(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, -1.0f), 0.0f));
}

TEST(TracingSink, DespawnMissesStationaryParticle) {
    const auto sink = TracingSink::builder()
                          .with_unit(make_static_plane_unit())
                          .build();

    // Zero speed can travel no distance, so the trace is skipped.
    EXPECT_FALSE(sink.despawn(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, 0.0f), 2.0f));
}

TEST(TracingSink, AdvanceMovesUnit) {
    auto sink = TracingSink::builder()
                    .with_unit(make_dynamic_plane_unit(Float3(0.0f, 0.0f, 1.0f)))
                    .build();

    sink.advance(0.5f);

    EXPECT_NEAR(sink.unit().sync().translation.z, 0.5f, tol);
}

TEST(TracingSink, DespawnMissesWithNegativeTimeStep) {
    const auto sink = TracingSink::builder()
                          .with_unit(make_static_plane_unit())
                          .build();

    // A non-positive dt covers no distance; the guard skips the trace entirely.
    EXPECT_FALSE(sink.despawn(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, -1.0f), -1.0f));
}

TEST(TracingSink, DespawnRejectsNaNVelocity) {
    const auto  sink = TracingSink::builder()
                          .with_unit(make_static_plane_unit())
                          .build();
    const float nan  = std::numeric_limits<float>::quiet_NaN();

    // A NaN speed fails the !(speed > 0) guard, so the particle is kept (no trace).
    EXPECT_FALSE(sink.despawn(Float3(0.0f, 0.0f, 1.0f), Float3(nan, 0.0f, 0.0f), 2.0f));
}
