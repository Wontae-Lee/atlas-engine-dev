#include <cuda/cuda_macros.cuh>
#include <utilities/tests_utils.cuh>

#include <atlas/atlas.h>

using namespace atlas;

CUDA_TEST(Collider, ConstructorStoresProvidedValues) {
    auto unit        = test::make_host_shared_unit<double>();
    auto interaction = test::make_host_shared_collider_surface_interaction<double>();
    HostBuffer<system::Unit<double>> units { *unit };
    HostBuffer<system::ColliderSurfaceInteraction<double>> interactions { *interaction };

    const system::Collider<double> collider(
        DeviceBuffer<system::Unit<double>>(units.begin(), units.end()),
        DeviceBuffer<system::ColliderSurfaceInteraction<double>>(interactions.begin(), interactions.end()));
    const auto host_interactions = ::atlas::test::cuda::to_host_vector(collider.surface_interactions());

    CUDA_EXPECT_EQ(collider.units().size(), 1u);
    CUDA_EXPECT_EQ(collider.surface_interactions().size(), 1u);
    CUDA_EXPECT_EQ(host_interactions[0].diffuse_sampling(), interaction->diffuse_sampling());
    CUDA_EXPECT_NEAR(host_interactions[0].restitution(), interaction->restitution(), eps);
}

CUDA_TEST(Collider, DefaultConstructorStartsEmpty) {
    system::Collider<double> collider;
    CUDA_EXPECT_TRUE(collider.units().empty());
    CUDA_EXPECT_TRUE(collider.surface_interactions().empty());
    CUDA_EXPECT_TRUE(collider.empty());
}

CUDA_TEST(Collider, BuilderBuildUsesProvidedValues) {
    auto unit        = test::make_host_shared_unit<double>();
    auto interaction = test::make_host_shared_collider_surface_interaction<double>();
    HostBuffer<system::Unit<double>> units { *unit };
    HostBuffer<system::ColliderSurfaceInteraction<double>> interactions { *interaction };

    const auto collider = system::Collider<double>::builder()
                              .with_units(units)
                              .with_surface_interactions(interactions)
                              .build();
    const auto host_interactions = ::atlas::test::cuda::to_host_vector(collider.surface_interactions());

    CUDA_EXPECT_EQ(collider.units().size(), 1u);
    CUDA_EXPECT_NEAR(host_interactions[0].restitution(), interaction->restitution(), eps);
}

CUDA_TEST(Collider, BuilderCreatesDefaultSurfaceInteractionWhenNotProvided) {
    auto unit = test::make_host_shared_unit<double>();
    HostBuffer<system::Unit<double>> units { *unit };

    const auto collider = system::Collider<double>::builder()
                              .with_units(units)
                              .build();
    const auto host_interactions = ::atlas::test::cuda::to_host_vector(collider.surface_interactions());

    CUDA_EXPECT_EQ(collider.units().size(), 1u);
    CUDA_EXPECT_EQ(collider.surface_interactions().size(), 1u);
    CUDA_EXPECT_EQ(host_interactions[0].diffuse_sampling(), system::DiffuseSampling::Uniform);
    CUDA_EXPECT_NEAR(host_interactions[0].restitution(), 1.0, eps);
    CUDA_EXPECT_NEAR(host_interactions[0].tangential_momentum_accommodation(), 1.0, eps);
}

CUDA_TEST(Collider, BuilderMakeHostSharedCreatesNonNullPointer) {
    auto unit        = test::make_host_shared_unit<double>();
    auto interaction = test::make_host_shared_collider_surface_interaction<double>();
    HostBuffer<system::Unit<double>> units { *unit };
    HostBuffer<system::ColliderSurfaceInteraction<double>> interactions { *interaction };

    const auto collider = system::Collider<double>::builder()
                              .with_units(units)
                              .with_surface_interactions(interactions)
                              .make_host_shared();
    const auto host_interactions = ::atlas::test::cuda::to_host_vector(collider->surface_interactions());

    CUDA_ASSERT_NE(collider, nullptr);
    CUDA_EXPECT_EQ(collider->units().size(), 1u);
    CUDA_EXPECT_NEAR(host_interactions[0].restitution(), interaction->restitution(), eps);
}

CUDA_TEST(Collider, BuilderThrowsWhenUnitsAreEmpty) {
    CUDA_EXPECT_THROW(
        (void)system::Collider<double>::builder().with_units({}),
        std::runtime_error);
}

CUDA_TEST(Collider, BuilderThrowsWhenSurfaceInteractionsAreEmpty) {
    auto unit = test::make_host_shared_unit<double>();
    HostBuffer<system::Unit<double>> units { *unit };

    CUDA_EXPECT_THROW(
        (void)system::Collider<double>::builder()
            .with_units(units)
            .with_surface_interactions({}),
        std::runtime_error);
}

CUDA_TEST(Collider, BuilderThrowsWhenBuildWithoutUnit) {
    CUDA_EXPECT_THROW(
        (void)system::Collider<double>::builder().build(),
        std::runtime_error);
}

CUDA_TEST(Collider, BuilderThrowsWhenSurfaceInteractionCountDoesNotMatchUnits) {
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

    CUDA_EXPECT_THROW(
        (void)system::Collider<double>::builder()
            .with_units(units)
            .with_surface_interactions(interactions)
            .build(),
        std::runtime_error);
}
