#include <atlas/source/surface_source.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace {

using atlas::Box;
using atlas::FluidPositionState;
using atlas::Geometry;
using atlas::Quaternion;
using atlas::SurfaceSource;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::Unit;
using atlas::Float3;
using atlas::tol;

SyncHostPtr
make_sync(const Float3& translation) {
    return Sync::builder()
        .with_rigid_pose(translation, Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
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
make_static_unit(const Float3& translation) {
    return Unit::builder()
        .with_geometry(make_unit_box())
        .with_sync(make_sync(translation))
        .build();
}

Unit
make_dynamic_unit(const Float3& velocity) {
    return Unit::builder()
        .with_geometry(make_unit_box())
        .with_sync(make_sync(Float3(0.0f, 0.0f, 0.0f)))
        .with_velocity(velocity)
        .build();
}

SurfaceSource
make_source(Unit unit) {
    return SurfaceSource::builder()
        .with_unit(std::move(unit))
        .with_tolerance(0.01f)
        .with_spacing(1.0f)
        .build();
}

}

TEST(SurfaceSource, BuilderRejectsMissingUnit) {
    EXPECT_THROW(
        static_cast<void>(SurfaceSource::builder().with_spacing(1.0f).build()),
        std::runtime_error);
}

TEST(SurfaceSource, BuilderRejectsNegativeTolerance) {
    EXPECT_THROW(
        static_cast<void>(SurfaceSource::builder().with_unit(make_static_unit(Float3(0.0f, 0.0f, 0.0f))).with_tolerance(-1.0f).build()),
        std::runtime_error);
}

TEST(SurfaceSource, BuilderRejectsNonPositiveSpacing) {
    EXPECT_THROW(
        static_cast<void>(SurfaceSource::builder().with_unit(make_static_unit(Float3(0.0f, 0.0f, 0.0f))).with_spacing(0.0f).build()),
        std::runtime_error);
}

TEST(SurfaceSource, CachesSurfaceSamples) {
    const auto source = make_source(make_static_unit(Float3(0.0f, 0.0f, 0.0f)));

    EXPECT_GT(source.cached_count(), std::size_t { 0 });
}

TEST(SurfaceSource, AdvanceMovesUnit) {
    auto source = make_source(make_dynamic_unit(Float3(0.0f, 0.0f, 1.0f)));

    source.advance(0.5f);

    EXPECT_NEAR(source.unit().sync().translation.z, 0.5f, tol);
}

TEST(SurfaceSource, SpawnWritesCachedParticlesAtOffset) {
    const auto        source = make_source(make_static_unit(Float3(5.0f, 0.0f, 0.0f)));
    const std::size_t count  = source.cached_count();
    ASSERT_GT(count, std::size_t { 0 });

    FluidPositionState positions(count + 8);
    const int          spawned = source.spawn(&positions, 0);

    EXPECT_EQ(static_cast<std::size_t>(spawned), count);

    // The unit sync translates local samples (x in [-1, 1]) to world x in [4, 6].
    const Float3 first = positions.data()[0];
    EXPECT_GE(first.x, 4.0f - tol);
    EXPECT_LE(first.x, 6.0f + tol);
}

TEST(SurfaceSource, SpawnRejectsNullTargetAndOutOfRangeOffset) {
    const auto         source = make_source(make_static_unit(Float3(0.0f, 0.0f, 0.0f)));
    FluidPositionState positions(4);

    EXPECT_EQ(source.spawn(nullptr, 0), 0);
    EXPECT_EQ(source.spawn(&positions, 100), 0);
}

TEST(SurfaceSource, DenserSpacingYieldsMoreSamples) {
    const auto coarse = SurfaceSource::builder()
                            .with_unit(make_static_unit(Float3(0.0f, 0.0f, 0.0f)))
                            .with_tolerance(0.01f)
                            .with_spacing(1.0f)
                            .build();
    const auto fine = SurfaceSource::builder()
                          .with_unit(make_static_unit(Float3(0.0f, 0.0f, 0.0f)))
                          .with_tolerance(0.01f)
                          .with_spacing(0.5f)
                          .build();

    EXPECT_GT(fine.cached_count(), coarse.cached_count());
}

TEST(SurfaceSource, MakeHostSharedBuildsCachedSource) {
    const auto source = SurfaceSource::builder()
                            .with_unit(make_static_unit(Float3(0.0f, 0.0f, 0.0f)))
                            .with_tolerance(0.01f)
                            .with_spacing(1.0f)
                            .make_host_shared();

    ASSERT_NE(source, nullptr);
    EXPECT_GT(source->cached_count(), std::size_t { 0 });
}

TEST(SurfaceSource, DefaultSourceCachesNothingAndSpawnsNothing) {
    const SurfaceSource source {};
    EXPECT_EQ(source.cached_count(), std::size_t { 0 });

    FluidPositionState positions(4);
    EXPECT_EQ(source.spawn(&positions, 0), 0);
}

TEST(SurfaceSource, SpawnTwiceAppendsAtSuccessiveOffsets) {
    const auto        source = make_source(make_static_unit(Float3(0.0f, 0.0f, 0.0f)));
    const std::size_t count  = source.cached_count();
    ASSERT_GT(count, std::size_t { 0 });

    FluidPositionState positions(2 * count);

    EXPECT_EQ(static_cast<std::size_t>(source.spawn(&positions, 0)), count);
    EXPECT_EQ(static_cast<std::size_t>(source.spawn(&positions, count)), count);
}
