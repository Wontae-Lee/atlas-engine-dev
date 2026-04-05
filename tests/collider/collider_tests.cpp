#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Collider, ConstructorStoresProvidedPointers) {
    auto unit        = test::make_host_shared_unit<double>();
    auto interaction = test::make_host_shared_collider_surface_interaction<double>();

    const system::Collider<double> collider(unit, interaction);

    EXPECT_EQ(collider.unit(), unit);
    EXPECT_EQ(collider.surface_interaction(), interaction);
}

TEST(Collider, SettersUpdateStoredPointers) {
    system::Collider<double> collider;

    auto unit        = test::make_host_shared_unit<double>();
    auto interaction = test::make_host_shared_collider_surface_interaction<double>();

    collider.set_unit(unit);
    collider.set_surface_interaction(interaction);

    EXPECT_EQ(collider.unit(), unit);
    EXPECT_EQ(collider.surface_interaction(), interaction);
}

TEST(Collider, BuilderBuildUsesProvidedPointers) {
    auto unit        = test::make_host_shared_unit<double>();
    auto interaction = test::make_host_shared_collider_surface_interaction<double>();

    const auto collider = system::Collider<double>::builder()
                              .with_unit(unit)
                              .with_surface_interaction(interaction)
                              .build();

    EXPECT_EQ(collider.unit(), unit);
    EXPECT_EQ(collider.surface_interaction(), interaction);
}

TEST(Collider, BuilderCreatesDefaultSurfaceInteractionWhenNotProvided) {
    auto unit = test::make_host_shared_unit<double>();

    const auto collider = system::Collider<double>::builder()
                              .with_unit(unit)
                              .build();

    ASSERT_NE(collider.surface_interaction(), nullptr);
    EXPECT_EQ(collider.unit(), unit);
    EXPECT_EQ(collider.surface_interaction()->diffuse_sampling(), system::DiffuseSampling::Uniform);
    EXPECT_NEAR(collider.surface_interaction()->restitution(), 1.0, eps);
    EXPECT_NEAR(collider.surface_interaction()->tangential_momentum_accommodation(), 1.0, eps);
}

TEST(Collider, BuilderMakeHostSharedCreatesNonNullPointer) {
    auto unit        = test::make_host_shared_unit<double>();
    auto interaction = test::make_host_shared_collider_surface_interaction<double>();

    const auto collider = system::Collider<double>::builder()
                              .with_unit(unit)
                              .with_surface_interaction(interaction)
                              .make_host_shared();

    ASSERT_NE(collider, nullptr);
    EXPECT_EQ(collider->unit(), unit);
    EXPECT_EQ(collider->surface_interaction(), interaction);
}

TEST(Collider, BuilderThrowsWhenUnitIsNull) {
    EXPECT_THROW(
        (void)system::Collider<double>::builder().with_unit(nullptr),
        std::runtime_error);
}

TEST(Collider, BuilderThrowsWhenSurfaceInteractionIsNull) {
    auto unit = test::make_host_shared_unit<double>();

    EXPECT_THROW(
        (void)system::Collider<double>::builder()
            .with_unit(unit)
            .with_surface_interaction(nullptr),
        std::runtime_error);
}

TEST(Collider, BuilderThrowsWhenBuildWithoutUnit) {
    EXPECT_THROW(
        (void)system::Collider<double>::builder().build(),
        std::runtime_error);
}

TEST(Collider, SettersThrowOnNullPointers) {
    system::Collider<double> collider;

    EXPECT_THROW((void)collider.set_unit(nullptr), std::runtime_error);
    EXPECT_THROW((void)collider.set_surface_interaction(nullptr), std::runtime_error);
}
