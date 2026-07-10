#include <atlas/collider/collider.h>

#include <atlas/collider/collider_type.h>
#include <atlas/collider/isothermal_collider.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

namespace {

using atlas::Collider;
using atlas::ColliderType;
using atlas::Geometry;
using atlas::HitSurface;
using atlas::IsothermalCollider;
using atlas::Plane;
using atlas::Quaternion;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::Unit;
using atlas::Float3;
using atlas::tol;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

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

IsothermalCollider
make_isothermal(const Float3& unit_velocity, const bool dynamic) {
    auto unit_builder = Unit::builder()
                            .with_geometry(make_plane_geometry())
                            .with_sync(make_identity_sync());

    const Unit unit = dynamic
        ? unit_builder.with_velocity(unit_velocity).build()
        : unit_builder.build();

    return IsothermalCollider::builder()
        .with_unit(unit)
        .with_momentum_accommodation_coefficient(0.0f)
        .with_restitution(1.0f)
        .build();
}

}

// The umbrella must remain a plain value so it can live in DeviceBuffer<Collider>.
static_assert(std::is_trivially_copyable_v<IsothermalCollider>,
              "IsothermalCollider must be trivially copyable for device buffers");
static_assert(std::is_trivially_copyable_v<Collider>,
              "Collider must be trivially copyable for device buffers");

TEST(Collider, DefaultConstructsIsothermal) {
    const Collider collider {};

    EXPECT_EQ(collider.type, ColliderType::isothermal);
}

TEST(Collider, WrapsIsothermalLeaf) {
    const Collider collider(make_isothermal(Float3(0.0f, 0.0f, 0.0f), false));

    EXPECT_EQ(collider.type, ColliderType::isothermal);
    EXPECT_NEAR(collider.isothermal.momentum_accommodation_coefficient(), 0.0f, tol);
}

TEST(Collider, TraceDispatchesToLeaf) {
    const Collider collider(make_isothermal(Float3(0.0f, 0.0f, 0.0f), false));

    const HitSurface hit = collider.trace(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, -1.0f), 2.0f);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 1.0f, tol);
}

TEST(Collider, CollideDispatchesToLeaf) {
    const Collider collider(make_isothermal(Float3(0.0f, 0.0f, 0.0f), false));

    Float3           position(0.0f, 0.0f, 1.0f);
    Float3           velocity(0.0f, 0.0f, -1.0f);
    const HitSurface hit = collider.trace(position, velocity, 2.0f);
    ASSERT_TRUE(hit.is_intersecting);

    collider.collide(hit, position, velocity, 2.0f);

    expect_vec_near(velocity, Float3(0.0f, 0.0f, 1.0f));
    expect_vec_near(position, Float3(0.0f, 0.0f, tol));
}

TEST(Collider, AdvanceDispatchesToLeaf) {
    Collider collider(make_isothermal(Float3(0.0f, 0.0f, 1.0f), true));

    collider.advance(0.5f);

    EXPECT_NEAR(collider.isothermal.unit().sync().translation.z, 0.5f, tol);
}

TEST(Collider, CopyPreservesBehaviour) {
    const Collider collider(make_isothermal(Float3(0.0f, 0.0f, 0.0f), false));
    const Collider copy = collider;

    EXPECT_EQ(copy.type, ColliderType::isothermal);

    const HitSurface hit = copy.trace(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, -1.0f), 2.0f);
    EXPECT_TRUE(hit.is_intersecting);
}
