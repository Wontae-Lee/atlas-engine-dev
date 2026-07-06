#include <atlas/collider/kernel/fast_collider_kernel.h>

#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <gtest/gtest.h>

namespace {

using atlas::FastColliderKernel;
using atlas::IsothermalSurfaceInteraction;
using atlas::Plane;
using atlas::Sync;
using atlas::Unit;
using atlas::Float3;
using atlas::tol;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

Unit
make_plane_unit(const Float3& linear_velocity  = Float3(0.0f, 0.0f, 0.0f),
                const Float3& angular_velocity = Float3(0.0f, 0.0f, 0.0f)) {
    static const auto geometry = Plane::builder()
                                     .with_point_normal(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 0.0f, 0.0f))
                                     .make_host_shared();
    const auto sync = Sync::builder()
                          .make_host_shared();

    auto builder = Unit::builder()
                       .with_geometry(atlas::Geometry(*geometry))
                       .with_sync(sync);

    if (linear_velocity.length_squared() > 0.0f) {
        builder.with_velocity(linear_velocity);
    }
    if (angular_velocity.length_squared() > 0.0f) {
        builder.with_angular_velocity(angular_velocity);
    }

    return builder.build();
}

atlas::SurfaceInteractionKernel
make_specular_interaction() {
    return atlas::SurfaceInteractionKernel(
        IsothermalSurfaceInteraction::builder()
            .with_restitution(1.0f)
            .with_momentum_acc(0.0f)
            .build());
}

}

TEST(FastColliderKernel, SurfaceVelocityCombinesLinearAndAngularMotion) {
    const auto unit = make_plane_unit(
        Float3(1.0f, 2.0f, 3.0f),
        Float3(0.0f, 0.0f, 2.0f));

    const auto velocity = FastColliderKernel::surface_velocity(
        unit,
        Float3(0.0f, 1.0f, 0.0f));

    expect_vec_near(velocity, Float3(-1.0f, 2.0f, 3.0f));
}

TEST(FastColliderKernel, SweepMotionUsesIncidentVelocityOverTimeStep) {
    const FastColliderKernel kernel;
    Float3 sweep_direction {};
    float sweep_speed {};
    float sweep_length {};

    kernel.sweep_motion(
        make_plane_unit(),
        Float3(0.0f, 0.0f, 0.0f),
        Float3(2.0f, 0.0f, 0.0f),
        2.0f,
        0.5f,
        sweep_direction,
        sweep_speed,
        sweep_length);

    expect_vec_near(sweep_direction, Float3(1.0f, 0.0f, 0.0f));
    EXPECT_NEAR(sweep_speed, 2.0f, tol);
    EXPECT_NEAR(sweep_length, 1.0f, tol);
}

TEST(FastColliderKernel, OperatorStopsAtHitPointAndUpdatesVelocity) {
    const FastColliderKernel kernel;
    Float3 position(-1.0f, 0.0f, 0.0f);
    Float3 velocity(2.0f, 0.0f, 0.0f);

    kernel(
        position,
        velocity,
        Float3(2.0f, 0.0f, 0.0f),
        Float3(0.0f, 0.0f, 0.0f),
        Float3(1.0f, 0.0f, 0.0f),
        1.0f,
        2.0f,
        1.0f,
        make_plane_unit(),
        make_specular_interaction());

    expect_vec_near(position, Float3(tol, 0.0f, 0.0f));
    expect_vec_near(velocity, Float3(-2.0f, 0.0f, 0.0f));
}
