#include <atlas/source/source.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/source/source_type.h>
#include <atlas/source/surface_source.h>
#include <atlas/source/volume_source.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace {

using atlas::Box;
using atlas::FluidPositionState;
using atlas::Geometry;
using atlas::Quaternion;
using atlas::Source;
using atlas::SourceType;
using atlas::SurfaceSource;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::Unit;
using atlas::VolumeSource;
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
make_unit(const Float3& translation, const Float3& velocity, const bool dynamic) {
    auto builder = Unit::builder()
                       .with_geometry(make_unit_box())
                       .with_sync(make_sync(translation));

    return dynamic ? builder.with_velocity(velocity).build() : builder.build();
}

SurfaceSource
make_surface_source() {
    return SurfaceSource::builder()
        .with_unit(make_unit(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), false))
        .with_tolerance(0.01f)
        .with_spacing(1.0f)
        .build();
}

VolumeSource
make_volume_source(const Float3& velocity, const bool dynamic) {
    return VolumeSource::builder()
        .with_unit(make_unit(Float3(0.0f, 0.0f, 0.0f), velocity, dynamic))
        .with_tolerance(0.0f)
        .with_spacing(1.0f)
        .build();
}

}

TEST(Source, DefaultConstructsSurface) {
    const Source source {};

    EXPECT_EQ(source.type, SourceType::surface);
}

TEST(Source, WrapsSurfaceLeaf) {
    const Source source(make_surface_source());

    EXPECT_EQ(source.type, SourceType::surface);
}

TEST(Source, WrapsVolumeLeaf) {
    const Source source(make_volume_source(Float3(0.0f, 0.0f, 0.0f), false));

    EXPECT_EQ(source.type, SourceType::volume);
}

TEST(Source, SpawnDispatchesToLeaf) {
    auto              leaf  = make_volume_source(Float3(0.0f, 0.0f, 0.0f), false);
    const std::size_t count = leaf.cached_count();
    ASSERT_GT(count, std::size_t { 0 });

    const Source       source(std::move(leaf));
    FluidPositionState positions(count + 8);

    EXPECT_EQ(static_cast<std::size_t>(source.spawn(&positions, 0)), count);
}

TEST(Source, AdvanceDispatchesToLeaf) {
    Source source(make_volume_source(Float3(0.0f, 0.0f, 1.0f), true));

    source.advance(0.5f);

    EXPECT_NEAR(source.volume.unit().sync().translation.z, 0.5f, tol);
}

TEST(Source, MoveConstructPreservesBehaviour) {
    auto              leaf  = make_volume_source(Float3(0.0f, 0.0f, 0.0f), false);
    const std::size_t count = leaf.cached_count();
    ASSERT_GT(count, std::size_t { 0 });

    Source       source(std::move(leaf));
    const Source moved = std::move(source);

    EXPECT_EQ(moved.type, SourceType::volume);

    FluidPositionState positions(count + 8);
    EXPECT_EQ(static_cast<std::size_t>(moved.spawn(&positions, 0)), count);
}

// The leaves own host-only device buffers, so the umbrella must be move-only.
TEST(Source, IsMoveOnlyNotCopyable) {
    EXPECT_FALSE(std::is_copy_constructible_v<Source>);
    EXPECT_FALSE(std::is_copy_assignable_v<Source>);
    EXPECT_TRUE(std::is_move_constructible_v<Source>);
    EXPECT_TRUE(std::is_move_assignable_v<Source>);
}

TEST(Source, MoveAssignReplacesActiveLeaf) {
    auto              leaf  = make_volume_source(Float3(0.0f, 0.0f, 0.0f), false);
    const std::size_t count = leaf.cached_count();
    ASSERT_GT(count, std::size_t { 0 });

    // Start on the default surface leaf, then move-assign a volume source over it.
    Source destination {};
    ASSERT_EQ(destination.type, SourceType::surface);

    destination = Source(std::move(leaf));

    EXPECT_EQ(destination.type, SourceType::volume);

    FluidPositionState positions(count + 8);
    EXPECT_EQ(static_cast<std::size_t>(destination.spawn(&positions, 0)), count);
}

TEST(Source, DefaultSurfaceLeafSpawnsNothing) {
    // A default umbrella holds an empty SurfaceSource, so dispatching spawn through
    // it writes nothing rather than reaching for an unbuilt cache.
    const Source       source {};
    FluidPositionState positions(4);

    ASSERT_EQ(source.type, SourceType::surface);
    EXPECT_EQ(source.spawn(&positions, 0), 0);
}
