#include <atlas/collider/isothermal_collider.h>

#include <atlas/collider/diffuse_sampling.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

using atlas::DiffuseSampling;
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

bool
is_finite_vec(const Float3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

SyncHostPtr
make_identity_sync() {
    return Sync::builder()
        .with_rigid_pose(Float3(0.0f, 0.0f, 0.0f), Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
        .make_host_shared();
}

// A z = 0 plane facing +z, so a particle descending along -z hits it at the origin.
Geometry
make_plane_geometry() {
    return Geometry(Plane(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f)));
}

Unit
make_static_unit() {
    return Unit::builder()
        .with_geometry(make_plane_geometry())
        .with_sync(make_identity_sync())
        .build();
}

Unit
make_dynamic_unit(const Float3& velocity) {
    return Unit::builder()
        .with_geometry(make_plane_geometry())
        .with_sync(make_identity_sync())
        .with_velocity(velocity)
        .build();
}

IsothermalCollider
make_collider(Unit unit,
              const float momentum_acc     = 0.0f,
              const float restitution       = 1.0f,
              const DiffuseSampling sampling = DiffuseSampling::uniform) {
    return IsothermalCollider::builder()
        .with_unit(std::move(unit))
        .with_momentum_accommodation_coefficient(momentum_acc)
        .with_restitution(restitution)
        .with_diffuse_sampling(sampling)
        .build();
}

}

TEST(IsothermalCollider, BuilderConfiguresState) {
    const auto collider = make_collider(make_static_unit(), 0.25f, 0.5f);

    EXPECT_NEAR(collider.momentum_accommodation_coefficient(), 0.25f, tol);
    EXPECT_FALSE(collider.unit().dynamic());
}

TEST(IsothermalCollider, BuilderRejectsMissingUnit) {
    EXPECT_THROW(
        static_cast<void>(IsothermalCollider::builder().build()),
        std::runtime_error);
}

TEST(IsothermalCollider, BuilderRejectsMomentumAccOutOfRange) {
    EXPECT_THROW(
        static_cast<void>(IsothermalCollider::builder().with_unit(make_static_unit()).with_momentum_accommodation_coefficient(1.5f).build()),
        std::runtime_error);
    EXPECT_THROW(
        static_cast<void>(IsothermalCollider::builder().with_unit(make_static_unit()).with_momentum_accommodation_coefficient(-0.1f).build()),
        std::runtime_error);
}

TEST(IsothermalCollider, BuilderRejectsNegativeRestitution) {
    EXPECT_THROW(
        static_cast<void>(IsothermalCollider::builder().with_unit(make_static_unit()).with_restitution(-1.0f).build()),
        std::runtime_error);
}

TEST(IsothermalCollider, MakeHostSharedBuildsCollider) {
    const auto collider = IsothermalCollider::builder()
                              .with_unit(make_static_unit())
                              .with_momentum_accommodation_coefficient(0.5f)
                              .make_host_shared();

    ASSERT_TRUE(static_cast<bool>(collider));
    EXPECT_NEAR(collider->momentum_accommodation_coefficient(), 0.5f, tol);
}

TEST(IsothermalCollider, ReflectIsSpecularWhenNotAccommodated) {
    const auto collider = make_collider(make_static_unit(), 0.0f, 1.0f);

    // momentum_acc == 0 => pure specular reflection about the surface normal.
    const Float3 reflected = collider.reflect(Float3(1.0f, 0.0f, -1.0f), Float3(0.0f, 0.0f, 1.0f));

    expect_vec_near(reflected, Float3(1.0f, 0.0f, 1.0f));
}

TEST(IsothermalCollider, ReflectScalesByRestitution) {
    const auto collider = make_collider(make_static_unit(), 0.0f, 0.5f);

    const Float3 reflected = collider.reflect(Float3(0.0f, 0.0f, -1.0f), Float3(0.0f, 0.0f, 1.0f));

    expect_vec_near(reflected, Float3(0.0f, 0.0f, 0.5f));
}

TEST(IsothermalCollider, ReflectDiffuseIsFinite) {
    const auto collider = make_collider(make_static_unit(), 1.0f, 1.0f, DiffuseSampling::cosine_weighted);

    const Float3 reflected = collider.reflect(Float3(0.5f, 0.25f, -1.0f), Float3(0.0f, 0.0f, 1.0f));

    EXPECT_TRUE(is_finite_vec(reflected));
}

TEST(IsothermalCollider, TraceHitsApproachingUnit) {
    const auto collider = make_collider(make_static_unit());

    const HitSurface hit = collider.trace(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, -1.0f), 2.0f);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 1.0f, tol);
    expect_vec_near(hit.point, Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(hit.normal, Float3(0.0f, 0.0f, 1.0f));
}

TEST(IsothermalCollider, TraceMissesRecedingParticle) {
    const auto collider = make_collider(make_static_unit());

    const HitSurface hit = collider.trace(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, 1.0f), 2.0f);

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(IsothermalCollider, TraceMissesZeroVelocity) {
    const auto collider = make_collider(make_static_unit());

    const HitSurface hit = collider.trace(Float3(0.0f, 0.0f, 1.0f), Float3(0.0f, 0.0f, 0.0f), 2.0f);

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(IsothermalCollider, TraceMissesWhenOutOfReach) {
    const auto collider = make_collider(make_static_unit());

    // Sweep length (speed * dt = 1) is shorter than the distance to the plane (5).
    const HitSurface hit = collider.trace(Float3(0.0f, 0.0f, 5.0f), Float3(0.0f, 0.0f, -1.0f), 1.0f);

    EXPECT_FALSE(hit.is_intersecting);
}

TEST(IsothermalCollider, CollideReflectsAndOffsetsPosition) {
    const auto collider = make_collider(make_static_unit(), 0.0f, 1.0f);

    Float3           position(0.0f, 0.0f, 1.0f);
    Float3           velocity(0.0f, 0.0f, -1.0f);
    const HitSurface hit = collider.trace(position, velocity, 2.0f);
    ASSERT_TRUE(hit.is_intersecting);

    collider.collide(hit, position, velocity, 2.0f);

    expect_vec_near(velocity, Float3(0.0f, 0.0f, 1.0f));
    expect_vec_near(position, Float3(0.0f, 0.0f, tol));
}

TEST(IsothermalCollider, CollideIsNoOpWithoutHit) {
    const auto collider = make_collider(make_static_unit());

    Float3           position(0.0f, 0.0f, 1.0f);
    Float3           velocity(0.0f, 0.0f, -1.0f);
    const HitSurface miss {};

    collider.collide(miss, position, velocity, 2.0f);

    expect_vec_near(position, Float3(0.0f, 0.0f, 1.0f));
    expect_vec_near(velocity, Float3(0.0f, 0.0f, -1.0f));
}

TEST(IsothermalCollider, AdvanceMovesDynamicUnit) {
    auto collider = make_collider(make_dynamic_unit(Float3(0.0f, 0.0f, 1.0f)));

    collider.advance(0.5f);

    EXPECT_NEAR(collider.unit().sync().translation.z, 0.5f, tol);
}
