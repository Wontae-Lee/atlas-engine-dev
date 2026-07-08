#include <atlas/sink/sink.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/math/math.h>
#include <atlas/sink/sink_type.h>
#include <atlas/sink/surface_sink.h>
#include <atlas/sink/tracing_sink.h>
#include <atlas/sink/volume_sink.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

namespace {

using atlas::Box;
using atlas::Geometry;
using atlas::Plane;
using atlas::Quaternion;
using atlas::Sink;
using atlas::SinkType;
using atlas::SurfaceSink;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::TracingSink;
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

Unit
make_box_unit(const Float3& velocity, const bool dynamic) {
    auto builder = Unit::builder()
                       .with_geometry(Geometry(Box::builder()
                                                   .with_lower_corner(Float3(-1.0f, -1.0f, -1.0f))
                                                   .with_upper_corner(Float3(1.0f, 1.0f, 1.0f))
                                                   .build()))
                       .with_sync(make_identity_sync());

    return dynamic ? builder.with_velocity(velocity).build() : builder.build();
}

Unit
make_plane_unit() {
    return Unit::builder()
        .with_geometry(Geometry(Plane(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f))))
        .with_sync(make_identity_sync())
        .build();
}

}

static_assert(std::is_trivially_copyable_v<Sink>,
              "Sink must be trivially copyable for device buffers");

TEST(Sink, DefaultConstructsSurface) {
    const Sink sink {};

    EXPECT_EQ(sink.type, SinkType::surface);
}

TEST(Sink, WrapsVolumeLeafAndDispatchesDespawn) {
    const Sink sink(VolumeSink::builder().with_unit(make_box_unit(Float3(0.0f, 0.0f, 0.0f), false)).build());

    EXPECT_EQ(sink.type, SinkType::volume);
    EXPECT_TRUE(sink.despawn(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));
    EXPECT_FALSE(sink.despawn(Float3(5.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));
}

TEST(Sink, WrapsSurfaceLeaf) {
    const Sink sink(SurfaceSink::builder()
                        .with_unit(make_box_unit(Float3(0.0f, 0.0f, 0.0f), false))
                        .with_tolerance(0.01f)
                        .build());

    EXPECT_EQ(sink.type, SinkType::surface);
    EXPECT_TRUE(sink.despawn(Float3(1.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));
}

TEST(Sink, WrapsTracingLeafAndDispatchesDespawn) {
    const Sink sink(TracingSink::builder().with_unit(make_plane_unit()).build());

    EXPECT_EQ(sink.type, SinkType::tracing);
    EXPECT_TRUE(sink.despawn(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, -1.0f), 2.0f));
    EXPECT_FALSE(sink.despawn(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, 1.0f), 2.0f));
}

TEST(Sink, AdvanceDispatchesToLeaf) {
    Sink sink(VolumeSink::builder().with_unit(make_box_unit(Float3(0.0f, 0.0f, 1.0f), true)).build());

    sink.advance(0.5f);

    EXPECT_NEAR(sink.volume.unit().sync().translation.z, 0.5f, tol);
}

TEST(Sink, CopyPreservesBehaviour) {
    const Sink sink(VolumeSink::builder().with_unit(make_box_unit(Float3(0.0f, 0.0f, 0.0f), false)).build());
    const Sink copy = sink;

    EXPECT_EQ(copy.type, SinkType::volume);
    EXPECT_TRUE(copy.despawn(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f), 0.0f));
}
