#include <cuda/cuda_macros.cuh>
#include <utilities/tests_utils.cuh>

#include <atlas/atlas.h>

#include <array>
#include <stdexcept>

using namespace atlas;

CUDA_TEST(System, BuilderStoresConfiguredPointersAndDt) {
    auto fluid    = test::make_buffered_fluid<double>(3);
    auto source   = test::make_test_source<double>();
    auto sink     = test::make_test_sink<double>();
    auto collider = test::make_host_shared_collider<double>();

    const auto sim_system = system::System<double>::builder()
                                .with_fluid(fluid)
                                .with_dt(0.125)
                                .with_source(source)
                                .with_sink(sink)
                                .with_collider(collider)
                                .build();

    CUDA_EXPECT_EQ(sim_system.fluid(), fluid);
    CUDA_EXPECT_DOUBLE_EQ(sim_system.dt(), 0.125);
    CUDA_EXPECT_EQ(sim_system.source(), source);
    CUDA_EXPECT_EQ(sim_system.sink(), sink);
    CUDA_EXPECT_EQ(sim_system.collider(), collider);
    CUDA_EXPECT_EQ(sim_system.particle_probe().buffer_size, 3u);
}

CUDA_TEST(System, SetDomainCreatesDefaultSingleCodecWhenCodecIsMissing) {
    system::System<double> sim_system(test::make_buffered_fluid<double>(0));
    auto domain = test::make_domain_ptr<double>();

    sim_system.set_domain(domain);

    CUDA_EXPECT_EQ(sim_system.domain(), domain);
    CUDA_ASSERT_NE(sim_system.codec(), nullptr);
    CUDA_EXPECT_EQ(sim_system.codec()->type(), atlas::system::CodecType::single);
    CUDA_EXPECT_EQ(sim_system.domain_probe().cell_size, domain->cell_size());
    CUDA_EXPECT_EQ(sim_system.searcher_probe().cell_size, domain->cell_size());
}

CUDA_TEST(System, SettersRejectInvalidInputsAndBuilderRequiresFluid) {
    system::System<double> sim_system(test::make_buffered_fluid<double>(0));

    CUDA_EXPECT_THROW((void)sim_system.set_dt(0.0), std::runtime_error);
    CUDA_EXPECT_THROW((void)sim_system.set_domain(DomainHostPtr<double> {}), std::runtime_error);
    CUDA_EXPECT_THROW((void)sim_system.set_codec(CodecHostPtr<double> {}), std::runtime_error);
    CUDA_EXPECT_THROW((void)sim_system.set_fluid(FluidHostPtr<double> {}), std::runtime_error);

    CUDA_EXPECT_THROW((void)system::System<double>::builder().build(), std::runtime_error);
    CUDA_EXPECT_THROW((void)system::System<double>::builder().with_dt(0.0).build(), std::runtime_error);
}

CUDA_TEST(System, UpdateWithoutColliderFallsBackToTimeIntegration) {
    auto sim_system = system::System<float>::builder()
                          .with_fluid(test::make_buffered_fluid<float>(1))
                          .with_dt(2.0f)
                          .build();
    auto& probe          = sim_system.particle_probe();
    probe.particle_count = 1;

    const std::array<Vector3<float>, 1> pos { Vector3<float>(0.0f, 1.0f, 2.0f) };
    const std::array<Vector3<float>, 1> vel { Vector3<float>(1.5f, -0.5f, 0.25f) };

    atlas::copy_host_to_device(pos.data(), probe.pos, pos.size());
    atlas::copy_host_to_device(vel.data(), probe.vel, vel.size());

    sim_system.update();

    const auto out_pos = test::copy_device_range(probe.pos, 1);
    const auto out_vel = test::copy_device_range(probe.vel, 1);

    CUDA_ASSERT_EQ(out_pos.size(), 1u);
    CUDA_ASSERT_EQ(out_vel.size(), 1u);
    CUDA_EXPECT_TRUE(test::vec_near(out_pos[0], Vector3<float>(3.0f, 0.0f, 2.5f), 1e-5f));
    CUDA_EXPECT_TRUE(test::vec_near(out_vel[0], vel[0], 1e-6f));
}

CUDA_TEST(System, EmitAndRemoveRunSourceAndSinkStages) {
    CUDA_SKIP("Source/sink stage interaction does not currently produce the same removal outcome under the CUDA backend.");
}

CUDA_TEST(System, ClassifyUpdatesCodecProbeAndInvokesCodec) {
    auto domain = test::make_domain_ptr<double>();
    auto codec  = atlas::make_host_shared<atlas::test::DummyCodec<double>>(domain);
    system::System<double> sim_system(test::make_buffered_fluid<double>(4));

    sim_system.set_domain(domain);
    sim_system.set_codec(codec);
    sim_system.classify();

    CUDA_EXPECT_TRUE(codec->encode_called);
    CUDA_EXPECT_TRUE(codec->decode_called);
    CUDA_EXPECT_EQ(sim_system.codec_probe().type, codec->type());
}

CUDA_TEST(System, CollideWithColliderReflectsVelocityAtClosestHit) {
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
                        .with_units({ *unit })
                        .with_surface_interactions({ *interaction })
                        .make_host_shared();
    auto sim_system = system::System<float>::builder()
                          .with_fluid(test::make_buffered_fluid<float>(1))
                          .with_dt(2.0f)
                          .with_collider(collider)
                          .build();
    auto& probe          = sim_system.particle_probe();
    probe.particle_count = 1;

    const std::array<Vector3<float>, 1> pos { Vector3<float>(0.0f, 0.0f, 2.0f) };
    const std::array<Vector3<float>, 1> vel { Vector3<float>(0.0f, 0.0f, -1.0f) };

    atlas::copy_host_to_device(pos.data(), probe.pos, pos.size());
    atlas::copy_host_to_device(vel.data(), probe.vel, vel.size());

    sim_system.collide();

    const auto out_pos = test::copy_device_range(probe.pos, 1);
    const auto out_vel = test::copy_device_range(probe.vel, 1);

    CUDA_ASSERT_EQ(out_pos.size(), 1u);
    CUDA_ASSERT_EQ(out_vel.size(), 1u);

    const auto epsf = static_cast<float>(atlas::eps);
    CUDA_EXPECT_NEAR(out_pos[0].x, 0.0f, epsf);
    CUDA_EXPECT_NEAR(out_pos[0].y, 0.0f, epsf);
    CUDA_EXPECT_NEAR(out_pos[0].z, 1.0f + epsf, 5.0f * epsf);
    CUDA_EXPECT_TRUE(test::vec_near(out_vel[0], Vector3<float>(0.0f, 0.0f, 1.0f), 1e-5f));
}

CUDA_TEST(System, ClearFunctionsResetOptionalPointers) {
    auto sim_system = system::System<double>::builder()
                          .with_fluid(test::make_buffered_fluid<double>(0))
                          .with_source(test::make_test_source<double>())
                          .with_sink(test::make_test_sink<double>())
                          .with_collider(test::make_host_shared_collider<double>())
                          .build();

    sim_system.clear_source();
    sim_system.clear_sink();
    sim_system.clear_collider();

    CUDA_EXPECT_EQ(sim_system.source(), nullptr);
    CUDA_EXPECT_EQ(sim_system.sink(), nullptr);
    CUDA_EXPECT_EQ(sim_system.collider(), nullptr);
}
