#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(ColliderSurfaceInteraction, DefaultConstructorSetsCanonicalParameters) {
    const ColliderSurfaceInteraction<double> interaction;

    EXPECT_EQ(interaction.diffuse_sampling(), system::DiffuseSampling::Uniform);
    EXPECT_NEAR(interaction.restitution(), 1.0, eps);
    EXPECT_NEAR(interaction.tangential_momentum_accommodation(), 1.0, eps);
    EXPECT_NEAR(interaction.temperature(), 273.15, eps);
}

TEST(ColliderSurfaceInteraction, SettersUpdateStoredParameters) {
    ColliderSurfaceInteraction<double> interaction;

    interaction.set_diffuse_sampling(system::DiffuseSampling::CosineWeighted);
    interaction.set_restitution(0.8);
    interaction.set_tangential_momentum_accommodation(0.25);
    interaction.set_temperature(350.0);

    EXPECT_EQ(interaction.diffuse_sampling(), system::DiffuseSampling::CosineWeighted);
    EXPECT_NEAR(interaction.restitution(), 0.8, eps);
    EXPECT_NEAR(interaction.tangential_momentum_accommodation(), 0.25, eps);
    EXPECT_NEAR(interaction.temperature(), 350.0, eps);
}

TEST(ColliderSurfaceInteraction, BuilderBuildStoresConfiguredValues) {
    const auto interaction = ColliderSurfaceInteraction<double>::builder()
                                 .with_diffuse_sampling(system::DiffuseSampling::CosineWeighted)
                                 .with_restitution(0.65)
                                 .with_tangential_momentum_accommodation(0.4)
                                 .with_temperature(420.0)
                                 .build();

    EXPECT_EQ(interaction.diffuse_sampling(), system::DiffuseSampling::CosineWeighted);
    EXPECT_NEAR(interaction.restitution(), 0.65, eps);
    EXPECT_NEAR(interaction.tangential_momentum_accommodation(), 0.4, eps);
    EXPECT_NEAR(interaction.temperature(), 420.0, eps);
}

TEST(ColliderSurfaceInteraction, BuilderMakeHostSharedCreatesConfiguredObject) {
    const auto interaction = ColliderSurfaceInteraction<double>::builder()
                                 .with_restitution(0.9)
                                 .with_tangential_momentum_accommodation(0.2)
                                 .with_temperature(300.0)
                                 .make_host_shared();

    ASSERT_NE(interaction, nullptr);
    EXPECT_NEAR(interaction->restitution(), 0.9, eps);
    EXPECT_NEAR(interaction->tangential_momentum_accommodation(), 0.2, eps);
    EXPECT_NEAR(interaction->temperature(), 300.0, eps);
}

TEST(ColliderSurfaceInteraction, BuilderRejectsInvalidPhysicalParameters) {
    EXPECT_THROW(
        ColliderSurfaceInteraction<double>::builder()
            .with_restitution(-0.1)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        ColliderSurfaceInteraction<double>::builder()
            .with_tangential_momentum_accommodation(1.1)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        ColliderSurfaceInteraction<double>::builder()
            .with_temperature(-1.0)
            .build(),
        std::runtime_error);
}

TEST(ColliderSurfaceInteraction, ZeroTmacProducesPureSpecularReflectionScaledByRestitution) {
    ColliderSurfaceInteraction<double> interaction;
    interaction.set_restitution(0.75);
    interaction.set_tangential_momentum_accommodation(0.0);

    const Vector3<double> incident(1.0, -2.0, -3.0);
    const Vector3<double> normal(0.0, 0.0, 1.0);
    const Vector3<double> expected = math::reflected(incident, normal) * 0.75;

    const Vector3<double> out = interaction(incident, normal);

    EXPECT_TRUE(test::vec_near(out, expected, 1e-12));
}

TEST(ColliderSurfaceInteraction, FullTmacProducesHemisphereDirectionWithRestitutionMagnitude) {
    ColliderSurfaceInteraction<double> interaction;
    interaction.set_diffuse_sampling(system::DiffuseSampling::CosineWeighted);
    interaction.set_restitution(0.5);
    interaction.set_tangential_momentum_accommodation(1.0);

    const Vector3<double> incident(0.3, -0.4, -2.0);
    const Vector3<double> normal(0.0, 0.0, 1.0);

    const Vector3<double> out = interaction(incident, normal);

    EXPECT_TRUE(test::is_finite_vec(out));
    EXPECT_GE(out.dot(normal), -eps);
    EXPECT_TRUE(test::vec_length_near(out, 0.5, 1e-9));
}
