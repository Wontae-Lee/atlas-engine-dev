#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Collider, ConstructorStoresProvidedValues) {
    auto unit        = test::make_host_shared_unit<double>();
    auto interaction = test::make_host_shared_collider_surface_interaction<double>();
    HostBuffer<system::Unit<double>> units { *unit };
    HostBuffer<system::ColliderSurfaceInteraction<double>> interactions { *interaction };

    const system::Collider<double> collider(
        DeviceBuffer<system::Unit<double>>(units.begin(), units.end()),
        DeviceBuffer<system::ColliderSurfaceInteraction<double>>(interactions.begin(), interactions.end()));

    EXPECT_EQ(collider.units().size(), 1u);
    EXPECT_EQ(collider.surface_interactions().size(), 1u);
    EXPECT_EQ(collider.surface_interactions()[0].diffuse_sampling(), interaction->diffuse_sampling());
    EXPECT_NEAR(collider.surface_interactions()[0].restitution(), interaction->restitution(), eps);
}

TEST(Collider, DefaultConstructorStartsEmpty) {
    system::Collider<double> collider;
    EXPECT_TRUE(collider.units().empty());
    EXPECT_TRUE(collider.surface_interactions().empty());
    EXPECT_TRUE(collider.empty());
}

TEST(Collider, BuilderBuildUsesProvidedValues) {
    auto unit        = test::make_host_shared_unit<double>();
    auto interaction = test::make_host_shared_collider_surface_interaction<double>();
    HostBuffer<system::Unit<double>> units { *unit };
    HostBuffer<system::ColliderSurfaceInteraction<double>> interactions { *interaction };

    const auto collider = system::Collider<double>::builder()
                              .with_units(units)
                              .with_surface_interactions(interactions)
                              .build();

    EXPECT_EQ(collider.units().size(), 1u);
    EXPECT_NEAR(collider.surface_interactions()[0].restitution(), interaction->restitution(), eps);
}

TEST(Collider, BuilderCreatesDefaultSurfaceInteractionWhenNotProvided) {
    auto unit = test::make_host_shared_unit<double>();
    HostBuffer<system::Unit<double>> units { *unit };

    const auto collider = system::Collider<double>::builder()
                              .with_units(units)
                              .build();

    EXPECT_EQ(collider.units().size(), 1u);
    EXPECT_EQ(collider.surface_interactions().size(), 1u);
    EXPECT_EQ(collider.surface_interactions()[0].diffuse_sampling(), system::DiffuseSampling::Uniform);
    EXPECT_NEAR(collider.surface_interactions()[0].restitution(), 1.0, eps);
    EXPECT_NEAR(collider.surface_interactions()[0].tangential_momentum_accommodation(), 1.0, eps);
}

TEST(Collider, BuilderMakeHostSharedCreatesNonNullPointer) {
    auto unit        = test::make_host_shared_unit<double>();
    auto interaction = test::make_host_shared_collider_surface_interaction<double>();
    HostBuffer<system::Unit<double>> units { *unit };
    HostBuffer<system::ColliderSurfaceInteraction<double>> interactions { *interaction };

    const auto collider = system::Collider<double>::builder()
                              .with_units(units)
                              .with_surface_interactions(interactions)
                              .make_host_shared();

    ASSERT_NE(collider, nullptr);
    EXPECT_EQ(collider->units().size(), 1u);
    EXPECT_NEAR(collider->surface_interactions()[0].restitution(), interaction->restitution(), eps);
}

TEST(Collider, BuilderThrowsWhenUnitsAreEmpty) {
    EXPECT_THROW(
        (void)system::Collider<double>::builder().with_units({}),
        std::runtime_error);
}

TEST(Collider, BuilderThrowsWhenSurfaceInteractionsAreEmpty) {
    auto unit = test::make_host_shared_unit<double>();
    HostBuffer<system::Unit<double>> units { *unit };

    EXPECT_THROW(
        (void)system::Collider<double>::builder()
            .with_units(units)
            .with_surface_interactions({}),
        std::runtime_error);
}

TEST(Collider, BuilderThrowsWhenBuildWithoutUnit) {
    EXPECT_THROW(
        (void)system::Collider<double>::builder().build(),
        std::runtime_error);
}

TEST(Collider, BuilderThrowsWhenSurfaceInteractionCountDoesNotMatchUnits) {
    auto unit0 = test::make_host_shared_unit<double>();
    auto unit1 = test::make_host_shared_unit<double>();
    auto interaction0 = test::make_host_shared_collider_surface_interaction<double>();
    auto interaction1 = test::make_host_shared_collider_surface_interaction<double>();

    HostBuffer<system::Unit<double>> units { *unit0, *unit1 };
    HostBuffer<system::ColliderSurfaceInteraction<double>> interactions {
        *interaction0,
        *interaction1,
        *interaction1
    };

    EXPECT_THROW(
        (void)system::Collider<double>::builder()
            .with_units(units)
            .with_surface_interactions(interactions)
            .build(),
        std::runtime_error);
}
