#include <atlas/collider/kernel/post_collider_kernel.h>

#include <atlas/collider/collider.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

namespace {

using atlas::Collider;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::FluidPositionState;
using atlas::FluidVelocityState;
using atlas::HostBuffer;
using atlas::IsothermalSurfaceInteraction;
using atlas::Plane;
using atlas::PostColliderKernel;
using atlas::PostColliderType;
using atlas::Sync;
using atlas::Unit;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Float3;
using atlas::tol;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

FluidHostPtr
make_fluid(const Float3& position, const Float3& velocity) {
    auto fluid = Fluid::builder()
                     .with_buffer_size(1)
                     .make_host_shared();

    fluid->set_particle_count(1);
    fluid->state<FluidPositionState>()->data()[0] = position;
    fluid->state<FluidVelocityState>()->data()[0] = velocity;

    return fluid;
}

UniverseHostPtr
make_universe(const HostBuffer<Unit>& collider_units) {
    return Universe::builder()
        .with_lower_corner(Float3(-10.0f, -10.0f, -10.0f))
        .with_upper_corner(Float3(10.0f, 10.0f, 10.0f))
        .with_cell_size(1.0f)
        .with_collider_units(collider_units)
        .make_host_shared();
}

Unit
make_plane_unit(const Float3& linear_velocity = Float3(0.0f, 0.0f, 0.0f)) {
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

    return builder.build();
}

IsothermalSurfaceInteraction
make_specular_interaction() {
    return IsothermalSurfaceInteraction::builder()
        .with_restitution(1.0f)
        .with_momentum_acc(0.0f)
        .build();
}

void
apply_kernel(PostColliderType type,
             Float3& position,
             Float3& velocity,
             const Unit& unit,
             float hit_distance,
             float sweep_speed) {
    const PostColliderKernel kernel(type);
    const auto interaction = atlas::SurfaceInteractionKernel(make_specular_interaction());

    kernel(
        position,
        velocity,
        Float3(2.0f, 0.0f, 0.0f),
        Float3(0.0f, 0.0f, 0.0f),
        Float3(1.0f, 0.0f, 0.0f),
        hit_distance,
        sweep_speed,
        1.0f,
        unit,
        interaction);
}

Float3
collide_position(PostColliderType type) {
    const auto fluid = make_fluid(
        Float3(-1.0f, 0.0f, 0.0f),
        Float3(2.0f, 0.0f, 0.0f));

    auto collider = Collider::builder()
                        .with_universe(make_universe(HostBuffer<Unit> {
                            make_plane_unit(Float3(0.5f, 0.0f, 0.0f))
                        }))
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<IsothermalSurfaceInteraction> { make_specular_interaction() })
                        .with_post_collider_type(type)
                        .build();

    collider.collide(1.0f);

    expect_vec_near(
        fluid->state<FluidVelocityState>()->data()[0],
        Float3(-1.0f, 0.0f, 0.0f));

    return fluid->state<FluidPositionState>()->data()[0];
}

}

TEST(PostColliderKernel, DefaultConstructsFastKernel) {
    const PostColliderKernel kernel;

    EXPECT_EQ(kernel.type, PostColliderType::fast);
}

TEST(PostColliderKernel, TypeConstructorSelectsRequestedKernel) {
    const PostColliderKernel kernel(PostColliderType::dt_remain);

    EXPECT_EQ(kernel.type, PostColliderType::dt_remain);
}

TEST(PostColliderKernel, CopyAndAssignmentPreserveActiveKernel) {
    const PostColliderKernel source(PostColliderType::precise);
    const PostColliderKernel copied(source);
    PostColliderKernel assigned;

    assigned = source;

    EXPECT_EQ(copied.type, PostColliderType::precise);
    EXPECT_EQ(assigned.type, PostColliderType::precise);
}

TEST(PostColliderKernel, FastStopsAtHitPointAndUpdatesVelocity) {
    Float3 position(-1.0f, 0.0f, 0.0f);
    Float3 velocity(2.0f, 0.0f, 0.0f);

    apply_kernel(
        PostColliderType::fast,
        position,
        velocity,
        make_plane_unit(),
        1.0f,
        2.0f);

    expect_vec_near(position, Float3(tol, 0.0f, 0.0f));
    expect_vec_near(velocity, Float3(-2.0f, 0.0f, 0.0f));
}

TEST(PostColliderKernel, DtRemainMovesForRemainingStepAfterVelocityUpdate) {
    Float3 position(-1.0f, 0.0f, 0.0f);
    Float3 velocity(2.0f, 0.0f, 0.0f);

    apply_kernel(
        PostColliderType::dt_remain,
        position,
        velocity,
        make_plane_unit(),
        1.0f,
        2.0f);

    expect_vec_near(position, Float3(-1.0f + tol, 0.0f, 0.0f));
    expect_vec_near(velocity, Float3(-2.0f, 0.0f, 0.0f));
}

TEST(PostColliderKernel, PreciseSweepUsesColliderSurfaceMotion) {
    const auto unit = make_plane_unit(Float3(0.5f, 0.0f, 0.0f));
    const PostColliderKernel kernel(PostColliderType::precise);

    Float3 sweep_direction {};
    float sweep_speed {};
    float sweep_length {};

    kernel.sweep_motion(
        unit,
        Float3(-1.0f, 0.0f, 0.0f),
        Float3(2.0f, 0.0f, 0.0f),
        2.0f,
        1.0f,
        sweep_direction,
        sweep_speed,
        sweep_length);

    expect_vec_near(sweep_direction, Float3(1.5f, 0.0f, 0.0f));
    EXPECT_NEAR(sweep_speed, 1.5f, tol);
    EXPECT_NEAR(sweep_length, 1.5f, tol);
}

TEST(Collider, PostColliderTypeControlsPostHitPosition) {
    const Float3 fast_position    = collide_position(PostColliderType::fast);
    const Float3 dt_position      = collide_position(PostColliderType::dt_remain);
    const Float3 precise_position = collide_position(PostColliderType::precise);

    EXPECT_NEAR(fast_position.x, tol, tol);
    EXPECT_NEAR(dt_position.x, -0.5f + tol, tol);
    EXPECT_NEAR(precise_position.x, -1.0f / 3.0f + tol, tol);
}
