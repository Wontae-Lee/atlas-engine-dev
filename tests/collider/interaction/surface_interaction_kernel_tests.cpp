#include <atlas/collider/interaction/surface_interaction_kernel.h>

#include <gtest/gtest.h>

namespace {

using atlas::FluidInternalEnergy;
using atlas::IsothermalSurfaceInteraction;
using atlas::MaterialProperties;
using atlas::MaxwellianSurfaceInteraction;
using atlas::SurfaceInteractionKernel;
using atlas::SurfaceInteractionType;
using atlas::Vector3;
using atlas::tol;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

}

TEST(SurfaceInteractionKernel, DefaultConstructsIsothermalInteraction) {
    const SurfaceInteractionKernel kernel;

    EXPECT_EQ(kernel.type, SurfaceInteractionType::isothermal);
}

TEST(SurfaceInteractionKernel, CopyAndAssignmentPreserveActiveInteraction) {
    const auto maxwellian = MaxwellianSurfaceInteraction::builder()
                                .with_temperature(400.0f)
                                .build();
    const SurfaceInteractionKernel source(maxwellian);
    const SurfaceInteractionKernel copied(source);
    SurfaceInteractionKernel assigned;

    assigned = source;

    EXPECT_EQ(copied.type, SurfaceInteractionType::maxwellian);
    EXPECT_EQ(assigned.type, SurfaceInteractionType::maxwellian);
}

TEST(SurfaceInteractionKernel, VelocityDispatchUsesActiveInteraction) {
    const auto isothermal = IsothermalSurfaceInteraction::builder()
                                .with_restitution(1.0f)
                                .with_momentum_acc(0.0f)
                                .build();
    const SurfaceInteractionKernel kernel(isothermal);

    const auto out = kernel(
        Vector3(1.0f, -2.0f, 0.0f),
        Vector3(0.0f, 1.0f, 0.0f));

    expect_vec_near(out, Vector3(1.0f, 2.0f, 0.0f));
}

TEST(SurfaceInteractionKernel, IsothermalInteractionPreservesInternalEnergy) {
    const auto isothermal = IsothermalSurfaceInteraction::builder()
                                .with_restitution(1.0f)
                                .with_momentum_acc(0.0f)
                                .build();
    const SurfaceInteractionKernel kernel(isothermal);
    const FluidInternalEnergy incident { 1.0f, 2.0f, 3.0f };
    const auto material = MaterialProperties::builder()
                              .with_mass(1.0f)
                              .with_molecular_mass(1.0f)
                              .build();

    const auto out = kernel.internal_energy(
        incident,
        Vector3(1.0f, 0.0f, 0.0f),
        Vector3(1.0f, 0.0f, 0.0f),
        material);

    EXPECT_NEAR(out.translational, incident.translational, tol);
    EXPECT_NEAR(out.rotational, incident.rotational, tol);
    EXPECT_NEAR(out.vibrational, incident.vibrational, tol);
}

TEST(SurfaceInteractionKernel, MaxwellianInteractionAppliesInternalEnergyAccommodation) {
    const auto maxwellian = MaxwellianSurfaceInteraction::builder()
                                .with_accommodation(1.0f, 1.0f, 1.0f, 1.0f)
                                .build();
    const SurfaceInteractionKernel kernel(maxwellian);
    const FluidInternalEnergy incident { 1.0f, 2.0f, 3.0f };
    const auto material = MaterialProperties::builder()
                              .with_mass(1.0f)
                              .with_molecular_mass(1.0f)
                              .with_rotational_dof(2)
                              .with_vibrational_dof(2)
                              .build();
    const auto expected = maxwellian.internal_energy(
        incident,
        Vector3(1.0f, 0.0f, 0.0f),
        Vector3(1.0f, 0.0f, 0.0f),
        material);

    const auto out = kernel.internal_energy(
        incident,
        Vector3(1.0f, 0.0f, 0.0f),
        Vector3(1.0f, 0.0f, 0.0f),
        material);

    EXPECT_NEAR(out.translational, expected.translational, tol);
    EXPECT_NEAR(out.rotational, expected.rotational, tol);
    EXPECT_NEAR(out.vibrational, expected.vibrational, tol);
}
