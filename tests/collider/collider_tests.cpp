#include "../utilities/test_utils.h"

#include <atlas/collider/collider.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/plane.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <testkit/testkit.h>

namespace {

using atlas::Box;
using atlas::Collider;
using atlas::ColliderSurfaceInteraction;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::HostBuffer;
using atlas::Plane;
using atlas::Sync;
using atlas::Unit;
using atlas::Vector3F;
using atlas::fluid::FluidPositionState;
using atlas::fluid::FluidVelocityState;
using atlas::test::vec_near;
using atlas::tol;

FluidHostPtr<float>
make_fluid() {
    return Fluid<float>::builder()
        .with_buffer_size(8)
        .make_host_shared();
}

Unit<float>
make_unit() {
    static const auto geometry = Box<float>::builder()
                                     .with_lower_corner(Vector3F(-1, -1, -1))
                                     .with_upper_corner(Vector3F(1, 1, 1))
                                     .make_host_shared();

    const auto sync = Sync<float>::builder()
                          .make_host_shared();

    return Unit<float>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .build();
}

Unit<float>
make_plane_unit(const Vector3F& linear_velocity = Vector3F(0, 0, 0),
                const Vector3F& angular_velocity = Vector3F(0, 0, 0)) {
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

    if (!vec_near(angular_velocity, Vector3F(0, 0, 0), 0.0f)) {
        builder.with_angular_velocity(angular_velocity);
    }

    return builder.build();
}

ColliderSurfaceInteraction<float>
make_interaction() {
    return ColliderSurfaceInteraction<float>::builder()
        .with_restitution(0.9f)
        .with_tangential_momentum_accommodation(0.2f)
        .build();
}

ColliderSurfaceInteraction<float>
make_specular_interaction() {
    return ColliderSurfaceInteraction<float>::builder()
        .with_restitution(1.0f)
        .with_tangential_momentum_accommodation(0.0f)
        .build();
}

} // namespace

TEST(Collider, EmptyReflectsMissingDependencies) {
    const Collider<float> empty_collider;

    EXPECT_TRUE(empty_collider.empty());
}

TEST(Collider, BuilderConstructsUsableCollider) {
    const auto fluid = make_fluid();

    const auto collider = Collider<float>::builder()
                              .with_units(HostBuffer<Unit<float>> { make_unit() })
                              .with_fluid(fluid)
                              .with_surface_interactions(
                                  HostBuffer<ColliderSurfaceInteraction<float>> { make_interaction() })
                              .build();

    EXPECT_FALSE(collider.empty());
}

TEST(Collider, BuilderRejectsInvalidConfiguration) {
    const auto fluid = make_fluid();

    EXPECT_THROW(
        Collider<float>::builder()
            .with_fluid(fluid)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        Collider<float>::builder()
            .with_units(HostBuffer<Unit<float>> { make_unit() })
            .build(),
        std::runtime_error);
}

TEST(Collider, MakeHostSharedBuildsCollider) {
    const auto fluid = make_fluid();

    const auto collider = Collider<float>::builder()
                              .with_units(HostBuffer<Unit<float>> { make_unit() })
                              .with_fluid(fluid)
                              .with_surface_interactions(
                                  HostBuffer<ColliderSurfaceInteraction<float>> { make_interaction() })
                              .make_host_shared();

    ASSERT_NE(collider, nullptr);
    EXPECT_FALSE(collider->empty());
}

TEST(Collider, UpdateAndCollideAreSafeNoOpsForDefaultFluidState) {
    const auto fluid = make_fluid();

    auto collider = Collider<float>::builder()
                        .with_units(HostBuffer<Unit<float>> { make_unit() })
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<ColliderSurfaceInteraction<float>> { make_interaction() })
                        .build();

    EXPECT_NO_THROW(collider.update(0.01f));
    EXPECT_NO_THROW(collider.collide(0.01f));
}

TEST(Collider, CollideAccountsForColliderLinearVelocityInSurfaceResponse) {
    
    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    auto* positions  = fluid->state<FluidPositionState<float>>();
    auto* velocities = fluid->state<FluidVelocityState<float>>();
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(velocities, nullptr);

    positions->data()[0]  = Vector3F(-1.0f, 0.0f, 0.0f);
    velocities->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);

    auto collider = Collider<float>::builder()
                        .with_units(HostBuffer<Unit<float>> {
                            make_plane_unit(Vector3F(0.0f, 2.0f, 0.0f))
                        })
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<ColliderSurfaceInteraction<float>> { make_specular_interaction() })
                        .build();

    collider.collide(1.0f);

    EXPECT_TRUE(vec_near(
        fluid->state<FluidVelocityState<float>>()->data()[0],
        Vector3F(-1.0f, 0.0f, 0.0f),
        tol));
}

TEST(Collider, CollideAccountsForColliderAngularVelocityAtContactPoint) {
    
    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    auto* positions  = fluid->state<FluidPositionState<float>>();
    auto* velocities = fluid->state<FluidVelocityState<float>>();
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(velocities, nullptr);

    positions->data()[0]  = Vector3F(-1.0f, 1.0f, 0.0f);
    velocities->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);

    auto collider = Collider<float>::builder()
                        .with_units(HostBuffer<Unit<float>> {
                            make_plane_unit(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.0f, 0.0f, 1.0f))
                        })
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<ColliderSurfaceInteraction<float>> { make_specular_interaction() })
                        .build();

    collider.collide(1.0f);

    EXPECT_TRUE(vec_near(
        fluid->state<FluidVelocityState<float>>()->data()[0],
        Vector3F(-3.0f, 0.0f, 0.0f),
        tol));
}
