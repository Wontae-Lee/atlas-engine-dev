#include "../utilities/tests_utils.h"

#include <atlas/collider/collider.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/plane.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <testkit/testkit.h>

namespace {

using T = float;

atlas::FluidHostPtr<T>
make_fluid() {
    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(8)
        .make_host_shared();
}

atlas::Unit<T>
make_unit() {
    const auto geometry = atlas::geometry::Box<T>::builder()
                              .with_lower_corner(atlas::Vector3<T>(-1, -1, -1))
                              .with_upper_corner(atlas::Vector3<T>(1, 1, 1))
                              .make_host_shared();

    const auto sync = atlas::physics::Sync<T>::builder()
                          .make_host_shared();

    return atlas::physics::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .build();
}

atlas::Unit<T>
make_plane_unit(const atlas::Vector3<T>& linear_velocity = atlas::Vector3<T>(0, 0, 0),
                const atlas::Vector3<T>& angular_velocity = atlas::Vector3<T>(0, 0, 0)) {
    const auto geometry = atlas::geometry::Plane<T>::builder()
                              .with_point_normal(atlas::Vector3<T>(0, 0, 0), atlas::Vector3<T>(1, 0, 0))
                              .make_host_shared();

    const auto sync = atlas::physics::Sync<T>::builder()
                          .make_host_shared();

    auto builder = atlas::physics::Unit<T>::builder()
                       .with_geometry(geometry)
                       .with_sync(sync);

    if (!atlas::test::vec_near(linear_velocity, atlas::Vector3<T>(0, 0, 0), static_cast<T>(0))) {
        builder.with_velocity(linear_velocity);
    }

    if (!atlas::test::vec_near(angular_velocity, atlas::Vector3<T>(0, 0, 0), static_cast<T>(0))) {
        builder.with_angular_velocity(angular_velocity);
    }

    return builder.build();
}

atlas::system::ColliderSurfaceInteraction<T>
make_interaction() {
    return atlas::system::ColliderSurfaceInteraction<T>::builder()
        .with_restitution(0.9f)
        .with_tangential_momentum_accommodation(0.2f)
        .build();
}

atlas::system::ColliderSurfaceInteraction<T>
make_specular_interaction() {
    return atlas::system::ColliderSurfaceInteraction<T>::builder()
        .with_restitution(1.0f)
        .with_tangential_momentum_accommodation(0.0f)
        .build();
}

} // namespace

TEST(Collider, EmptyReflectsMissingDependencies) {
    const atlas::system::Collider<T> empty_collider;

    EXPECT_TRUE(empty_collider.empty());
}

TEST(Collider, BuilderConstructsUsableCollider) {
    const auto fluid = make_fluid();

    const auto collider = atlas::system::Collider<T>::builder()
                              .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
                              .with_fluid(fluid)
                              .with_surface_interactions(
                                  atlas::HostBuffer<atlas::system::ColliderSurfaceInteraction<T>> { make_interaction() })
                              .build();

    EXPECT_FALSE(collider.empty());
}

TEST(Collider, BuilderRejectsInvalidConfiguration) {
    const auto fluid = make_fluid();

    EXPECT_THROW(
        atlas::system::Collider<T>::builder()
            .with_fluid(fluid)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::system::Collider<T>::builder()
            .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
            .build(),
        std::runtime_error);
}

TEST(Collider, MakeHostSharedBuildsCollider) {
    const auto fluid = make_fluid();

    const auto collider = atlas::system::Collider<T>::builder()
                              .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
                              .with_fluid(fluid)
                              .with_surface_interactions(
                                  atlas::HostBuffer<atlas::system::ColliderSurfaceInteraction<T>> { make_interaction() })
                              .make_host_shared();

    ASSERT_NE(collider, nullptr);
    EXPECT_FALSE(collider->empty());
}

TEST(Collider, UpdateAndCollideAreSafeNoOpsForDefaultFluidState) {
    const auto fluid = make_fluid();

    auto collider = atlas::system::Collider<T>::builder()
                        .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            atlas::HostBuffer<atlas::system::ColliderSurfaceInteraction<T>> { make_interaction() })
                        .build();

    EXPECT_NO_THROW(collider.update(0.01f));
    EXPECT_NO_THROW(collider.collide(0.01f));
}

TEST(Collider, CollideAccountsForColliderLinearVelocityInSurfaceResponse) {
    using Vec3 = atlas::Vector3<T>;

    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    auto* positions  = fluid->state<atlas::fluid::FluidPositionState<T>>();
    auto* velocities = fluid->state<atlas::fluid::FluidVelocityState<T>>();
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(velocities, nullptr);

    positions->data()[0]  = Vec3(-1.0f, 0.0f, 0.0f);
    velocities->data()[0] = Vec3(1.0f, 0.0f, 0.0f);

    auto collider = atlas::system::Collider<T>::builder()
                        .with_units(atlas::HostBuffer<atlas::Unit<T>> {
                            make_plane_unit(Vec3(0.0f, 2.0f, 0.0f))
                        })
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            atlas::HostBuffer<atlas::system::ColliderSurfaceInteraction<T>> { make_specular_interaction() })
                        .build();

    collider.collide(1.0f);

    EXPECT_TRUE(atlas::test::vec_near(
        fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[0],
        Vec3(-1.0f, 0.0f, 0.0f),
        static_cast<T>(1e-5)));
}

TEST(Collider, CollideAccountsForColliderAngularVelocityAtContactPoint) {
    using Vec3 = atlas::Vector3<T>;

    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    auto* positions  = fluid->state<atlas::fluid::FluidPositionState<T>>();
    auto* velocities = fluid->state<atlas::fluid::FluidVelocityState<T>>();
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(velocities, nullptr);

    positions->data()[0]  = Vec3(-1.0f, 1.0f, 0.0f);
    velocities->data()[0] = Vec3(1.0f, 0.0f, 0.0f);

    auto collider = atlas::system::Collider<T>::builder()
                        .with_units(atlas::HostBuffer<atlas::Unit<T>> {
                            make_plane_unit(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f))
                        })
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            atlas::HostBuffer<atlas::system::ColliderSurfaceInteraction<T>> { make_specular_interaction() })
                        .build();

    collider.collide(1.0f);

    EXPECT_TRUE(atlas::test::vec_near(
        fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[0],
        Vec3(-3.0f, 0.0f, 0.0f),
        static_cast<T>(1e-5)));
}
