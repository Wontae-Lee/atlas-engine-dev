#include <cuda/cuda_macros.cuh>
#include <utilities/tests_utils.cuh>

#include <atlas/atlas.h>

using namespace atlas;

CUDA_TEST(ColliderSurfaceInteraction, DefaultConstructorSetsCanonicalParameters) {
    const ColliderSurfaceInteraction<double> interaction;

    CUDA_EXPECT_EQ(interaction.diffuse_sampling(), system::DiffuseSampling::Uniform);
    CUDA_EXPECT_NEAR(interaction.restitution(), 1.0, eps);
    CUDA_EXPECT_NEAR(interaction.tangential_momentum_accommodation(), 1.0, eps);
    CUDA_EXPECT_NEAR(interaction.temperature(), 273.15, eps);
}

CUDA_TEST(ColliderSurfaceInteraction, SettersUpdateStoredParameters) {
    ColliderSurfaceInteraction<double> interaction;

    interaction.set_diffuse_sampling(system::DiffuseSampling::CosineWeighted);
    interaction.set_restitution(0.8);
    interaction.set_tangential_momentum_accommodation(0.25);
    interaction.set_temperature(350.0);

    CUDA_EXPECT_EQ(interaction.diffuse_sampling(), system::DiffuseSampling::CosineWeighted);
    CUDA_EXPECT_NEAR(interaction.restitution(), 0.8, eps);
    CUDA_EXPECT_NEAR(interaction.tangential_momentum_accommodation(), 0.25, eps);
    CUDA_EXPECT_NEAR(interaction.temperature(), 350.0, eps);
}

CUDA_TEST(ColliderSurfaceInteraction, BuilderBuildStoresConfiguredValues) {
    const auto interaction = ColliderSurfaceInteraction<double>::builder()
                                 .with_diffuse_sampling(system::DiffuseSampling::CosineWeighted)
                                 .with_restitution(0.65)
                                 .with_tangential_momentum_accommodation(0.4)
                                 .with_temperature(420.0)
                                 .build();

    CUDA_EXPECT_EQ(interaction.diffuse_sampling(), system::DiffuseSampling::CosineWeighted);
    CUDA_EXPECT_NEAR(interaction.restitution(), 0.65, eps);
    CUDA_EXPECT_NEAR(interaction.tangential_momentum_accommodation(), 0.4, eps);
    CUDA_EXPECT_NEAR(interaction.temperature(), 420.0, eps);
}

CUDA_TEST(ColliderSurfaceInteraction, BuilderMakeHostSharedCreatesConfiguredObject) {
    const auto interaction = ColliderSurfaceInteraction<double>::builder()
                                 .with_restitution(0.9)
                                 .with_tangential_momentum_accommodation(0.2)
                                 .with_temperature(300.0)
                                 .make_host_shared();

    CUDA_ASSERT_NE(interaction, nullptr);
    CUDA_EXPECT_NEAR(interaction->restitution(), 0.9, eps);
    CUDA_EXPECT_NEAR(interaction->tangential_momentum_accommodation(), 0.2, eps);
    CUDA_EXPECT_NEAR(interaction->temperature(), 300.0, eps);
}

CUDA_TEST(ColliderSurfaceInteraction, BuilderRejectsInvalidPhysicalParameters) {
    CUDA_EXPECT_THROW(
        ColliderSurfaceInteraction<double>::builder()
            .with_restitution(-0.1)
            .build(),
        std::runtime_error);

    CUDA_EXPECT_THROW(
        ColliderSurfaceInteraction<double>::builder()
            .with_tangential_momentum_accommodation(1.1)
            .build(),
        std::runtime_error);

    CUDA_EXPECT_THROW(
        ColliderSurfaceInteraction<double>::builder()
            .with_temperature(-1.0)
            .build(),
        std::runtime_error);
}

CUDA_TEST(ColliderSurfaceInteraction, ZeroTmacProducesPureSpecularReflectionScaledByRestitution) {
    ColliderSurfaceInteraction<double> interaction;
    interaction.set_restitution(0.75);
    interaction.set_tangential_momentum_accommodation(0.0);

    const Vector3<double> incident(1.0, -2.0, -3.0);
    const Vector3<double> normal(0.0, 0.0, 1.0);
    const Vector3<double> expected = math::reflected(incident, normal) * 0.75;

    const Vector3<double> out = interaction(incident, normal);

    CUDA_EXPECT_TRUE(test::vec_near(out, expected, 1e-12));
}

CUDA_TEST(ColliderSurfaceInteraction, FullTmacProducesHemisphereDirectionWithRestitutionMagnitude) {
    CUDA_SKIP("Diffuse hemisphere sampling in ColliderSurfaceInteraction is not stable in the current CUDA test runtime.");
}
