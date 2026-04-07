#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

using namespace atlas;

namespace {

template <typename T>
atlas::FluidHostPtr<T>
make_test_fluid() {
    const auto species = atlas::system::FluidicParticle<T>::builder()
                             .with_molecular_mass(T(1))
                             .make_host_shared();

    return atlas::system::Fluid<T>::builder()
        .add_species(species, T(1))
        .make_host_shared();
}

}

TEST(System, AddColliderStoresPointer) {
    system::System<double> sim_system(0);
    auto collider = test::make_host_shared_collider<double>();

    sim_system.add_collider(collider);

    ASSERT_EQ(sim_system.colliders().size(), 1u);
    EXPECT_EQ(sim_system.colliders().front(), collider);
}

TEST(System, AddColliderValueOverloadCreatesStoredCollider) {
    system::System<double> sim_system(0);
    auto collider_ptr = test::make_host_shared_collider<double>();

    sim_system.add_collider(*collider_ptr);

    ASSERT_EQ(sim_system.colliders().size(), 1u);
    ASSERT_NE(sim_system.colliders().front(), nullptr);
    EXPECT_EQ(sim_system.colliders().front()->unit(), collider_ptr->unit());
    EXPECT_EQ(sim_system.colliders().front()->surface_interaction(), collider_ptr->surface_interaction());
}

TEST(System, SetCollidersCopiesConfiguredPointerListAndClearRemovesAll) {
    system::System<double> sim_system(0);
    HostBuffer<ColliderHostPtr<double>> colliders {
        test::make_host_shared_collider<double>(),
        test::make_host_shared_collider<double>()
    };

    sim_system.set_colliders(colliders);

    ASSERT_EQ(sim_system.colliders().size(), 2u);
    EXPECT_EQ(sim_system.colliders()[0], colliders[0]);
    EXPECT_EQ(sim_system.colliders()[1], colliders[1]);

    sim_system.clear_colliders();
    EXPECT_TRUE(sim_system.colliders().empty());
}

TEST(System, SetCollidersCopiesConfiguredValueList) {
    system::System<double> sim_system(0);
    HostBuffer<Collider<double>> colliders {
        *test::make_host_shared_collider<double>(),
        *test::make_host_shared_collider<double>()
    };

    sim_system.set_colliders(colliders);

    ASSERT_EQ(sim_system.colliders().size(), 2u);
    ASSERT_NE(sim_system.colliders()[0], nullptr);
    ASSERT_NE(sim_system.colliders()[1], nullptr);
    EXPECT_EQ(sim_system.colliders()[0]->unit(), colliders[0].unit());
    EXPECT_EQ(sim_system.colliders()[1]->surface_interaction(), colliders[1].surface_interaction());
}

TEST(System, BuilderBuildStoresConfiguredColliders) {
    auto collider0 = test::make_host_shared_collider<double>();
    auto collider1 = test::make_host_shared_collider<double>();

    const auto sim_system = system::System<double>::builder()
                                .with_buffer_size(0)
                                .with_collider(collider0)
                                .with_collider(collider1)
                                .build();

    ASSERT_EQ(sim_system.colliders().size(), 2u);
    EXPECT_EQ(sim_system.colliders()[0], collider0);
    EXPECT_EQ(sim_system.colliders()[1], collider1);
}

TEST(System, BuilderStoresConfiguredDtSourcesAndSinks) {
    auto source = atlas::Source<double>::builder()
                      .with_unit(*test::make_host_shared_unit<double>())
                      .with_fluid(make_test_fluid<double>())
                      .make_host_shared();
    auto sink = atlas::Sink<double>::builder()
                    .with_unit(*test::make_host_shared_unit<double>())
                    .make_host_shared();

    const auto sim_system = system::System<double>::builder()
                                .with_buffer_size(3)
                                .with_dt(0.125)
                                .with_source(source)
                                .with_sink(sink)
                                .build();

    EXPECT_DOUBLE_EQ(sim_system.dt(), 0.125);
    ASSERT_EQ(sim_system.sources().size(), 1u);
    ASSERT_EQ(sim_system.sinks().size(), 1u);
    EXPECT_EQ(sim_system.sources().front(), source);
    EXPECT_EQ(sim_system.sinks().front(), sink);
    EXPECT_EQ(sim_system.particle_probe().buffer_size, 3u);
}

TEST(System, BuilderStoresConfiguredDomainAndCodec) {
    auto domain = test::make_domain_ptr<double>();
    auto codec = atlas::make_host_shared<atlas::test::DummyCodec<double>>(domain);

    const auto sim_system = system::System<double>::builder()
                                .with_buffer_size(0)
                                .with_domain(domain)
                                .with_codec(codec)
                                .build();

    EXPECT_EQ(sim_system.domain(), domain);
    EXPECT_EQ(sim_system.codec(), codec);
}

TEST(System, BuilderUsesSingleCodecByDefaultWhenDomainIsConfigured) {
    auto domain = test::make_domain_ptr<double>();

    const auto sim_system = system::System<double>::builder()
                                .with_buffer_size(0)
                                .with_domain(domain)
                                .build();

    ASSERT_NE(sim_system.codec(), nullptr);
    EXPECT_EQ(sim_system.codec()->type(), atlas::system::CodecType::single);
}

TEST(System, SetDomainAndCodecStoreConfiguredPointers) {
    system::System<double> sim_system(0);
    auto domain = test::make_domain_ptr<double>();
    auto codec = atlas::make_host_shared<atlas::test::DummyCodec<double>>(domain);

    sim_system.set_domain(domain);
    sim_system.set_codec(codec);

    EXPECT_EQ(sim_system.domain(), domain);
    EXPECT_EQ(sim_system.codec(), codec);
}

TEST(System, SetDomainCreatesDefaultSingleCodecWhenCodecIsMissing) {
    system::System<double> sim_system(0);
    auto domain = test::make_domain_ptr<double>();

    sim_system.set_domain(domain);

    ASSERT_NE(sim_system.codec(), nullptr);
    EXPECT_EQ(sim_system.codec()->type(), atlas::system::CodecType::single);
}

TEST(System, StoresRuntimeProbesAsMembers) {
    auto domain = test::make_domain_ptr<double>();

    auto sim_system = system::System<double>::builder()
                          .with_buffer_size(4)
                          .with_domain(domain)
                          .build();

    EXPECT_EQ(sim_system.particle_probe().buffer_size, 4u);
    EXPECT_EQ(sim_system.domain_probe().cell_size, domain->cell_size());
    EXPECT_EQ(sim_system.searcher_probe().cell_size, domain->cell_size());
    EXPECT_EQ(sim_system.codec_probe().type, atlas::system::CodecType::single);
}

TEST(System, BuilderAcceptsColliderValuesInHostBuffer) {
    HostBuffer<Collider<double>> colliders {
        *test::make_host_shared_collider<double>(),
        *test::make_host_shared_collider<double>()
    };

    const auto sim_system = system::System<double>::builder()
                                .with_buffer_size(0)
                                .with_colliders(colliders)
                                .build();

    ASSERT_EQ(sim_system.colliders().size(), 2u);
    ASSERT_NE(sim_system.colliders()[0], nullptr);
    ASSERT_NE(sim_system.colliders()[1], nullptr);
    EXPECT_EQ(sim_system.colliders()[0]->unit(), colliders[0].unit());
    EXPECT_EQ(sim_system.colliders()[1]->surface_interaction(), colliders[1].surface_interaction());
}

TEST(System, BuilderMakeHostSharedCreatesNonNullPointer) {
    auto collider = test::make_host_shared_collider<double>();

    const auto sim_system = system::System<double>::builder()
                                .with_buffer_size(4)
                                .with_collider(collider)
                                .make_host_shared();

    ASSERT_NE(sim_system, nullptr);
    ASSERT_EQ(sim_system->colliders().size(), 1u);
    EXPECT_EQ(sim_system->colliders().front(), collider);
    EXPECT_EQ(sim_system->particle_probe().buffer_size, 4);
}

TEST(System, BuilderAndSetterAllowNullColliderPointersAndSkipThem) {
    system::System<double> sim_system(0);
    sim_system.add_collider(ColliderHostPtr<double> {});
    EXPECT_EQ(sim_system.colliders().size(), 1u);

    const auto built = system::System<double>::builder()
                           .with_buffer_size(0)
                           .with_collider(ColliderHostPtr<double> {})
                           .build();
    EXPECT_EQ(built.colliders().size(), 1u);

    HostBuffer<ColliderHostPtr<double>> colliders { ColliderHostPtr<double> {} };
    sim_system.set_colliders(colliders);
    EXPECT_EQ(sim_system.colliders().size(), 1u);

    const auto built_from_buffer = system::System<double>::builder()
                                       .with_buffer_size(0)
                                       .with_colliders(colliders)
                                       .build();
    EXPECT_EQ(built_from_buffer.colliders().size(), 1u);
}

TEST(System, BuilderAndSetterAllowNullSourcesAndSinksAndRejectOnlyInvalidDtDomainCodec) {
    system::System<double> sim_system(0);

    EXPECT_THROW((void)sim_system.set_dt(0.0), std::runtime_error);
    EXPECT_THROW((void)sim_system.set_domain(DomainHostPtr<double> {}), std::runtime_error);
    EXPECT_THROW((void)sim_system.set_codec(CodecHostPtr<double> {}), std::runtime_error);
    EXPECT_THROW((void)system::System<double>::builder().with_dt(0.0).build(), std::runtime_error);
    EXPECT_THROW((void)system::System<double>::builder().with_domain(DomainHostPtr<double> {}), std::runtime_error);
    EXPECT_THROW((void)system::System<double>::builder().with_codec(CodecHostPtr<double> {}), std::runtime_error);

    HostBuffer<SourceHostPtr<double>> sources { SourceHostPtr<double> {} };
    HostBuffer<SinkHostPtr<double>> sinks { SinkHostPtr<double> {} };
    sim_system.add_source(SourceHostPtr<double> {});
    sim_system.add_sink(SinkHostPtr<double> {});
    sim_system.set_sources(sources);
    sim_system.set_sinks(sinks);
    EXPECT_EQ(sim_system.sources().size(), 1u);
    EXPECT_EQ(sim_system.sinks().size(), 1u);

    const auto built = system::System<double>::builder()
                           .with_buffer_size(0)
                           .with_source(SourceHostPtr<double> {})
                           .with_sink(SinkHostPtr<double> {})
                           .with_sources(sources)
                           .with_sinks(sinks)
                           .build();
    EXPECT_EQ(built.sources().size(), 1u);
    EXPECT_EQ(built.sinks().size(), 1u);
}

TEST(System, UpdateSkipsNullSourceSinkAndColliderStages) {
    auto sim_system = system::System<float>::builder()
                          .with_buffer_size(1)
                          .with_dt(1.0f)
                          .with_source(SourceHostPtr<float> {})
                          .with_sink(SinkHostPtr<float> {})
                          .with_collider(ColliderHostPtr<float> {})
                          .build();
    auto& probe = sim_system.particle_probe();
    probe.particle_count = 1;

    const std::array<Vector3<float>, 1> pos { Vector3<float>(0.0f, 0.0f, 0.0f) };
    const std::array<Vector3<float>, 1> vel { Vector3<float>(1.0f, 2.0f, 3.0f) };

    atlas::copy_host_to_device(pos.data(), probe.pos, pos.size());
    atlas::copy_host_to_device(vel.data(), probe.vel, vel.size());

    sim_system.update();

    const auto out_pos = test::copy_device_range(probe.pos, 1);
    const auto out_vel = test::copy_device_range(probe.vel, 1);

    EXPECT_TRUE(test::vec_near(out_pos[0], Vector3<float>(1.0f, 2.0f, 3.0f), 1e-6f));
    EXPECT_TRUE(test::vec_near(out_vel[0], vel[0], 1e-6f));
}

TEST(System, OperatorWithoutCollidersFallsBackToTimeIntegration) {
    auto sim_system = system::System<float>::builder()
                          .with_buffer_size(1)
                          .with_dt(2.0f)
                          .build();
    auto& probe = sim_system.particle_probe();
    probe.particle_count = 1;

    const std::array<Vector3<float>, 1> pos { Vector3<float>(0.0f, 1.0f, 2.0f) };
    const std::array<Vector3<float>, 1> vel { Vector3<float>(1.5f, -0.5f, 0.25f) };

    atlas::copy_host_to_device(pos.data(), probe.pos, pos.size());
    atlas::copy_host_to_device(vel.data(), probe.vel, vel.size());

    sim_system.advect();

    const auto out_pos = test::copy_device_range(probe.pos, 1);
    const auto out_vel = test::copy_device_range(probe.vel, 1);

    ASSERT_EQ(out_pos.size(), 1u);
    ASSERT_EQ(out_vel.size(), 1u);
    EXPECT_TRUE(test::vec_near(out_pos[0], Vector3<float>(3.0f, 0.0f, 2.5f), 1e-5f));
    EXPECT_TRUE(test::vec_near(out_vel[0], vel[0], 1e-6f));
}

TEST(System, NoArgOperatorUsesStoredDt) {
    auto sim_system = system::System<float>::builder()
                          .with_buffer_size(1)
                          .with_dt(2.0f)
                          .build();
    auto& probe = sim_system.particle_probe();

    const std::array<Vector3<float>, 1> pos { Vector3<float>(0.0f, 1.0f, 2.0f) };
    const std::array<Vector3<float>, 1> vel { Vector3<float>(1.5f, -0.5f, 0.25f) };

    probe.particle_count = 1;
    atlas::copy_host_to_device(pos.data(), probe.pos, pos.size());
    atlas::copy_host_to_device(vel.data(), probe.vel, vel.size());

    sim_system.update();

    const auto out_pos = test::copy_device_range(probe.pos, 1);
    const auto out_vel = test::copy_device_range(probe.vel, 1);

    ASSERT_EQ(out_pos.size(), 1u);
    ASSERT_EQ(out_vel.size(), 1u);
    EXPECT_TRUE(test::vec_near(out_pos[0], Vector3<float>(3.0f, 0.0f, 2.5f), 1e-5f));
    EXPECT_TRUE(test::vec_near(out_vel[0], vel[0], 1e-6f));
}

TEST(System, EmitAndRemoveRunSourceAndSinkStages) {
    auto source = atlas::Source<float>::builder()
                      .with_unit(*test::make_host_shared_unit<float>())
                      .with_fluid(make_test_fluid<float>())
                      .with_spacing(1.0f)
                      .make_host_shared();
    auto sink = atlas::Sink<float>::builder()
                    .with_unit(*test::make_host_shared_unit<float>())
                    .with_despawn_type(system::DespawnType::Volume)
                    .make_host_shared();

    auto sim_system = system::System<float>::builder()
                          .with_buffer_size(128)
                          .with_source(source)
                          .with_sink(sink)
                          .build();
    sim_system.particle_probe().particle_count = 0;

    EXPECT_EQ(sim_system.particle_probe().particle_count, 0);

    sim_system.emit();
    EXPECT_GT(sim_system.particle_probe().particle_count, 0);

    sim_system.remove();
    EXPECT_EQ(sim_system.particle_probe().particle_count, 0);
}

TEST(System, OperatorWithColliderReflectsVelocityAtClosestHit) {
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

    auto sim_system = system::System<float>::builder()
                          .with_buffer_size(1)
                          .with_dt(2.0f)
                          .with_collider(collider)
                          .build();
    auto& probe = sim_system.particle_probe();
    probe.particle_count = 1;

    const std::array<Vector3<float>, 1> pos { Vector3<float>(0.0f, 0.0f, 2.0f) };
    const std::array<Vector3<float>, 1> vel { Vector3<float>(0.0f, 0.0f, -1.0f) };

    atlas::copy_host_to_device(pos.data(), probe.pos, pos.size());
    atlas::copy_host_to_device(vel.data(), probe.vel, vel.size());

    sim_system.advect();

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
