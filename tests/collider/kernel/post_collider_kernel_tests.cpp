#include "../../utilities/test_utils.h"

#include <atlas/collider/collider.h>
#include <atlas/collider/kernel/post_collider_kernel.h>
#include <atlas/geometry/plane.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <testkit/testkit.h>

namespace {

using atlas::Collider;
using atlas::IsothermalSurfaceInteraction;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::HostBuffer;
using atlas::Plane;
using atlas::PostColliderKernel;
using atlas::PostColliderType;
using atlas::Sync;
using atlas::Unit;
using atlas::Vector3F;
using atlas::FluidPositionState;
using atlas::FluidVelocityState;
using atlas::test::vec_near;
using atlas::tol;

FluidHostPtr<float>
make_fluid(const Vector3F& position, const Vector3F& velocity) {
    auto fluid = Fluid<float>::builder()
                     .with_buffer_size(1)
                     .make_host_shared();

    fluid->set_particle_count(1);
    fluid->state<FluidPositionState<float>>()->data()[0] = position;
    fluid->state<FluidVelocityState<float>>()->data()[0] = velocity;

    return fluid;
}

Unit<float>
make_plane_unit(const Vector3F& linear_velocity = Vector3F(0, 0, 0)) {
    static const auto geometry = Plane<float>::builder()
                                     .with_point_normal(Vector3F(0, 0, 0), Vector3F(1, 0, 0))
                                     .make_host_shared();

    const auto sync = Sync<float>::builder()
                          .make_host_shared();

    auto builder = Unit<float>::builder()
                       .with_geometry(geometry)
                       .with_sync(sync);

    if (!vec_near(linear_velocity, Vector3F(0, 0, 0), 0.0f)) {
        builder.with_velocity(linear_velocity);
    }

    return builder.build();
}

IsothermalSurfaceInteraction<float>
make_specular_interaction() {
    return IsothermalSurfaceInteraction<float>::builder()
        .with_restitution(1.0f)
        .with_momentum_acc(0.0f)
        .build();
}

void
apply_kernel(PostColliderType type,
             Vector3F& position,
             Vector3F& velocity,
             const Unit<float>& unit,
             float hit_distance,
             float sweep_speed) {
    const PostColliderKernel<float> kernel(type);
    const auto interaction = make_specular_interaction();

    kernel(
        position,
        velocity,
        Vector3F(2.0f, 0.0f, 0.0f),
        Vector3F(0.0f, 0.0f, 0.0f),
        Vector3F(1.0f, 0.0f, 0.0f),
        hit_distance,
        sweep_speed,
        1.0f,
        unit,
        interaction);
}

Vector3F
collide_position(PostColliderType type) {
    const auto fluid = make_fluid(
        Vector3F(-1.0f, 0.0f, 0.0f),
        Vector3F(2.0f, 0.0f, 0.0f));

    auto collider = Collider<float>::builder()
                        .with_units(HostBuffer<Unit<float>> {
                            make_plane_unit(Vector3F(0.5f, 0.0f, 0.0f))
                        })
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<IsothermalSurfaceInteraction<float>> { make_specular_interaction() })
                        .with_post_collider_type(type)
                        .build();

    collider.collide(1.0f);

    EXPECT_TRUE(vec_near(
        fluid->state<FluidVelocityState<float>>()->data()[0],
        Vector3F(-1.0f, 0.0f, 0.0f),
        tol));

    return fluid->state<FluidPositionState<float>>()->data()[0];
}

} // namespace

TEST(PostColliderKernel, DefaultConstructsFastKernel) {
    const PostColliderKernel<float> kernel;

    EXPECT_EQ(kernel.type, PostColliderType::fast);
}

TEST(PostColliderKernel, TypeConstructorSelectsRequestedKernel) {
    const PostColliderKernel<float> kernel(PostColliderType::dt_remain);

    EXPECT_EQ(kernel.type, PostColliderType::dt_remain);
}

TEST(PostColliderKernel, CopyAndAssignmentPreserveActiveKernel) {
    const PostColliderKernel<float> source(PostColliderType::precise);
    const PostColliderKernel<float> copied(source);
    PostColliderKernel<float> assigned;

    assigned = source;

    EXPECT_EQ(copied.type, PostColliderType::precise);
    EXPECT_EQ(assigned.type, PostColliderType::precise);
}

TEST(PostColliderKernel, DestroyActiveAndCopyFromRebuildActiveKernel) {
    const PostColliderKernel<float> source(PostColliderType::dt_remain);
    PostColliderKernel<float> kernel(PostColliderType::fast);

    kernel.destroy_active();
    kernel.copy_from(source);

    EXPECT_EQ(kernel.type, PostColliderType::dt_remain);
}

TEST(PostColliderKernel, FastStopsAtHitPointAndUpdatesVelocity) {
    Vector3F position(-1.0f, 0.0f, 0.0f);
    Vector3F velocity(2.0f, 0.0f, 0.0f);

    apply_kernel(
        PostColliderType::fast,
        position,
        velocity,
        make_plane_unit(),
        1.0f,
        2.0f);

    EXPECT_TRUE(vec_near(position, Vector3F(static_cast<float>(tol), 0.0f, 0.0f), tol));
    EXPECT_TRUE(vec_near(velocity, Vector3F(-2.0f, 0.0f, 0.0f), tol));
}

TEST(PostColliderKernel, DtRemainMovesForRemainingStepAfterVelocityUpdate) {
    Vector3F position(-1.0f, 0.0f, 0.0f);
    Vector3F velocity(2.0f, 0.0f, 0.0f);

    apply_kernel(
        PostColliderType::dt_remain,
        position,
        velocity,
        make_plane_unit(),
        1.0f,
        2.0f);

    EXPECT_TRUE(vec_near(position, Vector3F(-1.0f + static_cast<float>(tol), 0.0f, 0.0f), tol));
    EXPECT_TRUE(vec_near(velocity, Vector3F(-2.0f, 0.0f, 0.0f), tol));
}

TEST(PostColliderKernel, PreciseSweepUsesColliderSurfaceMotion) {
    const auto unit = make_plane_unit(Vector3F(0.5f, 0.0f, 0.0f));
    const PostColliderKernel<float> kernel(PostColliderType::precise);

    Vector3F sweep_direction {};
    float sweep_speed {};
    float sweep_length {};

    kernel.sweep_motion(
        unit,
        Vector3F(-1.0f, 0.0f, 0.0f),
        Vector3F(2.0f, 0.0f, 0.0f),
        2.0f,
        1.0f,
        sweep_direction,
        sweep_speed,
        sweep_length);

    EXPECT_TRUE(vec_near(sweep_direction, Vector3F(1.5f, 0.0f, 0.0f), tol));
    EXPECT_NEAR(sweep_speed, 1.5f, tol);
    EXPECT_NEAR(sweep_length, 1.5f, tol);
}

TEST(Collider, PostColliderTypeControlsPostHitPosition) {
    const Vector3F fast_position = collide_position(PostColliderType::fast);
    const Vector3F dt_position   = collide_position(PostColliderType::dt_remain);
    const Vector3F precise_position = collide_position(PostColliderType::precise);

    EXPECT_NEAR(fast_position.x, static_cast<float>(tol), tol);
    EXPECT_NEAR(dt_position.x, -0.5f + static_cast<float>(tol), tol);
    EXPECT_NEAR(precise_position.x, -1.0f / 3.0f + static_cast<float>(tol), tol);
}
