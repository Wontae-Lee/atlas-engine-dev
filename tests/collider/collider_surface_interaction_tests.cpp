#include "../utilities/tests_utils.h"

#include <atlas/collider/collider_surface_interaction.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(ColliderSurfaceInteraction, DefaultStateIsWellDefined) {
    const atlas::system::ColliderSurfaceInteraction<T> interaction;

    EXPECT_EQ(interaction.diffuse_sampling(), atlas::system::DiffuseSampling::Uniform);
    EXPECT_NEAR(interaction.restitution(), 1.0f, kEps);
    EXPECT_NEAR(interaction.tangential_momentum_accommodation(), 1.0f, kEps);
    EXPECT_NEAR(interaction.temperature(), 273.15f, 1e-3f);
}

TEST(ColliderSurfaceInteraction, SettersUpdateState) {
    atlas::system::ColliderSurfaceInteraction<T> interaction;

    interaction.set_diffuse_sampling(atlas::system::DiffuseSampling::CosineWeighted);
    interaction.set_restitution(0.75f);
    interaction.set_tangential_momentum_accommodation(0.25f);
    interaction.set_temperature(350.0f);

    EXPECT_EQ(interaction.diffuse_sampling(), atlas::system::DiffuseSampling::CosineWeighted);
    EXPECT_NEAR(interaction.restitution(), 0.75f, kEps);
    EXPECT_NEAR(interaction.tangential_momentum_accommodation(), 0.25f, kEps);
    EXPECT_NEAR(interaction.temperature(), 350.0f, kEps);
}

TEST(ColliderSurfaceInteraction, BuilderConstructsConfiguredInteraction) {
    const auto interaction = atlas::system::ColliderSurfaceInteraction<T>::builder()
                                 .with_diffuse_sampling(atlas::system::DiffuseSampling::CosineWeighted)
                                 .with_restitution(0.5f)
                                 .with_tangential_momentum_accommodation(0.0f)
                                 .with_temperature(400.0f)
                                 .build();

    EXPECT_EQ(interaction.diffuse_sampling(), atlas::system::DiffuseSampling::CosineWeighted);
    EXPECT_NEAR(interaction.restitution(), 0.5f, kEps);
    EXPECT_NEAR(interaction.tangential_momentum_accommodation(), 0.0f, kEps);
    EXPECT_NEAR(interaction.temperature(), 400.0f, kEps);
}

TEST(ColliderSurfaceInteraction, BuilderRejectsInvalidParameters) {
    EXPECT_THROW(
        atlas::system::ColliderSurfaceInteraction<T>::builder()
            .with_restitution(-1.0f)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::system::ColliderSurfaceInteraction<T>::builder()
            .with_tangential_momentum_accommodation(2.0f)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::system::ColliderSurfaceInteraction<T>::builder()
            .with_temperature(-1.0f)
            .build(),
        std::runtime_error);
}

TEST(ColliderSurfaceInteraction, SpecularModeMatchesReflectedDirection) {
    atlas::system::ColliderSurfaceInteraction<T> interaction;
    interaction.set_restitution(0.5f);
    interaction.set_tangential_momentum_accommodation(0.0f);

    const Vec3 incident(1.0f, -2.0f, 0.0f);
    const Vec3 normal(0.0f, 1.0f, 0.0f);

    const Vec3 out = interaction(incident, normal);
    const Vec3 expected = atlas::math::reflected(incident, normal) * 0.5f;

    EXPECT_TRUE(atlas::test::vec_near(out, expected, 1e-4f));
}

TEST(ColliderSurfaceInteraction, DiffuseModeReturnsFiniteDirection) {
    atlas::system::ColliderSurfaceInteraction<T> interaction;
    interaction.set_tangential_momentum_accommodation(1.0f);

    const Vec3 out = interaction(Vec3(1.0f, -1.0f, 0.5f), Vec3(0.0f, 1.0f, 0.0f));

    EXPECT_TRUE(atlas::test::is_finite_vec(out));
    EXPECT_GT(out.length(), 0.0f);
}
