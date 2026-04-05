#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

using namespace atlas;

TEST(Advector, AddColliderStoresPointer) {
    system::Advector<double> advector;
    auto collider = test::make_host_shared_collider<double>();

    advector.add_collider(collider);

    ASSERT_EQ(advector.colliders().size(), 1u);
    EXPECT_EQ(advector.colliders().front(), collider);
}

TEST(Advector, AddColliderValueOverloadCreatesStoredCollider) {
    system::Advector<double> advector;
    auto collider_ptr = test::make_host_shared_collider<double>();

    advector.add_collider(*collider_ptr);

    ASSERT_EQ(advector.colliders().size(), 1u);
    ASSERT_NE(advector.colliders().front(), nullptr);
    EXPECT_EQ(advector.colliders().front()->unit(), collider_ptr->unit());
    EXPECT_EQ(advector.colliders().front()->surface_interaction(), collider_ptr->surface_interaction());
}

TEST(Advector, SetCollidersCopiesConfiguredListAndClearRemovesAll) {
    system::Advector<double> advector;
    HostBuffer<ColliderHostPtr<double>> colliders {
        test::make_host_shared_collider<double>(),
        test::make_host_shared_collider<double>()
    };

    advector.set_colliders(colliders);

    ASSERT_EQ(advector.colliders().size(), 2u);
    EXPECT_EQ(advector.colliders()[0], colliders[0]);
    EXPECT_EQ(advector.colliders()[1], colliders[1]);

    advector.clear_colliders();
    EXPECT_TRUE(advector.colliders().empty());
}

TEST(Advector, BuilderBuildStoresConfiguredColliders) {
    auto collider0 = test::make_host_shared_collider<double>();
    auto collider1 = test::make_host_shared_collider<double>();

    const auto advector = system::Advector<double>::builder()
                              .with_collider(collider0)
                              .with_collider(collider1)
                              .build();

    ASSERT_EQ(advector.colliders().size(), 2u);
    EXPECT_EQ(advector.colliders()[0], collider0);
    EXPECT_EQ(advector.colliders()[1], collider1);
}

TEST(Advector, BuilderMakeHostSharedCreatesNonNullPointer) {
    auto collider = test::make_host_shared_collider<double>();

    const auto advector = system::Advector<double>::builder()
                              .with_collider(collider)
                              .make_host_shared();

    ASSERT_NE(advector, nullptr);
    ASSERT_EQ(advector->colliders().size(), 1u);
    EXPECT_EQ(advector->colliders().front(), collider);
}

TEST(Advector, BuilderAndSetterRejectNullColliderPointers) {
    system::Advector<double> advector;

    EXPECT_THROW((void)advector.add_collider(ColliderHostPtr<double> {}), std::runtime_error);
    EXPECT_THROW((void)system::Advector<double>::builder().with_collider(ColliderHostPtr<double> {}), std::runtime_error);

    HostBuffer<ColliderHostPtr<double>> colliders { ColliderHostPtr<double> {} };
    EXPECT_THROW((void)advector.set_colliders(colliders), std::runtime_error);
    EXPECT_THROW((void)system::Advector<double>::builder().with_colliders(colliders), std::runtime_error);
}

TEST(Advector, OperatorWithoutCollidersFallsBackToTimeIntegration) {
    system::ParticleData<float> particle_data(1);
    auto probe = particle_data.make_device_probe();

    const std::array<Vector3<float>, 1> pos { Vector3<float>(0.0f, 1.0f, 2.0f) };
    const std::array<Vector3<float>, 1> vel { Vector3<float>(1.5f, -0.5f, 0.25f) };

    atlas::copy_host_to_device(pos.data(), probe.pos, pos.size());
    atlas::copy_host_to_device(vel.data(), probe.vel, vel.size());

    system::Advector<float> advector;
    advector(probe, 2.0f);

    const auto out_pos = test::copy_device_range(probe.pos, 1);
    const auto out_vel = test::copy_device_range(probe.vel, 1);

    ASSERT_EQ(out_pos.size(), 1u);
    ASSERT_EQ(out_vel.size(), 1u);
    EXPECT_TRUE(test::vec_near(out_pos[0], Vector3<float>(3.0f, 0.0f, 2.5f), 1e-5f));
    EXPECT_TRUE(test::vec_near(out_vel[0], vel[0], 1e-6f));
}

TEST(Advector, OperatorWithColliderReflectsVelocityAtClosestHit) {
    auto geometry = atlas::make_host_shared<geometry::Sphere<float>>(
        geometry::Sphere<float>(Vector3<float>(0.0f, 0.0f, 0.0f), 1.0f));
    auto sync = atlas::make_host_shared<system::Sync<float>>();
    auto unit = system::Unit<float>::builder()
                    .with_geometry(geometry)
                    .with_sync(sync)
                    .make_host_shared();
    auto interaction = atlas::ColliderSurfaceInteraction<float>::builder()
                           .with_restitution(1.0f)
                           .with_tangential_momentum_accommodation(0.0f)
                           .make_host_shared();
    auto collider = atlas::Collider<float>::builder()
                        .with_unit(unit)
                        .with_surface_interaction(interaction)
                        .make_host_shared();

    system::ParticleData<float> particle_data(1);
    auto probe = particle_data.make_device_probe();

    const std::array<Vector3<float>, 1> pos { Vector3<float>(0.0f, 0.0f, 2.0f) };
    const std::array<Vector3<float>, 1> vel { Vector3<float>(0.0f, 0.0f, -1.0f) };

    atlas::copy_host_to_device(pos.data(), probe.pos, pos.size());
    atlas::copy_host_to_device(vel.data(), probe.vel, vel.size());

    auto advector = system::Advector<float>::builder()
                        .with_collider(collider)
                        .build();

    advector(probe, 2.0f);

    const auto out_pos = test::copy_device_range(probe.pos, 1);
    const auto out_vel = test::copy_device_range(probe.vel, 1);

    ASSERT_EQ(out_pos.size(), 1u);
    ASSERT_EQ(out_vel.size(), 1u);

    const float epsf = static_cast<float>(atlas::eps);
    EXPECT_NEAR(out_pos[0].x, 0.0f, epsf);
    EXPECT_NEAR(out_pos[0].y, 0.0f, epsf);
    EXPECT_NEAR(out_pos[0].z, 1.0f + epsf, 5.0f * epsf);
    EXPECT_TRUE(test::vec_near(out_vel[0], Vector3<float>(0.0f, 0.0f, 1.0f), 1e-5f));
}
