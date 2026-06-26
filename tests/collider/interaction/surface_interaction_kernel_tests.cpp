#include "../../utilities/test_utils.h"

#include <atlas/collider/interaction/surface_interaction_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::FluidInternalEnergy;
using atlas::IsothermalSurfaceInteraction;
using atlas::MaterialProperties;
using atlas::MaxwellianSurfaceInteraction;
using atlas::SurfaceInteractionKernel;
using atlas::SurfaceInteractionType;
using atlas::Vector3F;
using atlas::tol;
using atlas::test::vec_near;

} // namespace

TEST(SurfaceInteractionKernel, DefaultConstructsIsothermalInteraction) {
    const SurfaceInteractionKernel<float> kernel;

    EXPECT_EQ(kernel.type, SurfaceInteractionType::isothermal);
}

TEST(SurfaceInteractionKernel, CopyAndAssignmentPreserveActiveInteraction) {
    const auto maxwellian = MaxwellianSurfaceInteraction<float>::builder()
                                .with_temperature(400.0f)
                                .build();
    const SurfaceInteractionKernel<float> source(maxwellian);
    const SurfaceInteractionKernel<float> copied(source);
    SurfaceInteractionKernel<float> assigned;

    assigned = source;

    EXPECT_EQ(copied.type, SurfaceInteractionType::maxwellian);
    EXPECT_EQ(assigned.type, SurfaceInteractionType::maxwellian);
}

TEST(SurfaceInteractionKernel, DestroyActiveAndCopyFromRebuildActiveInteraction) {
    const auto maxwellian = MaxwellianSurfaceInteraction<float>::builder()
                                .with_temperature(400.0f)
                                .build();
    const SurfaceInteractionKernel<float> source(maxwellian);
    SurfaceInteractionKernel<float> kernel;

    kernel.destroy_active();
    kernel.copy_from(source);

    EXPECT_EQ(kernel.type, SurfaceInteractionType::maxwellian);
}

TEST(SurfaceInteractionKernel, VelocityDispatchUsesActiveInteraction) {
    const auto isothermal = IsothermalSurfaceInteraction<float>::builder()
                                .with_restitution(1.0f)
                                .with_momentum_acc(0.0f)
                                .build();
    const SurfaceInteractionKernel<float> kernel(isothermal);

    const auto out = kernel(
        Vector3F(1.0f, -2.0f, 0.0f),
        Vector3F(0.0f, 1.0f, 0.0f));

    EXPECT_TRUE(vec_near(out, Vector3F(1.0f, 2.0f, 0.0f), tol));
}

TEST(SurfaceInteractionKernel, IsothermalInteractionPreservesInternalEnergy) {
    const auto isothermal = IsothermalSurfaceInteraction<float>::builder()
                                .with_restitution(1.0f)
                                .with_momentum_acc(0.0f)
                                .build();
    const SurfaceInteractionKernel<float> kernel(isothermal);
    const FluidInternalEnergy<float> incident { 1.0f, 2.0f, 3.0f };
    const auto material = MaterialProperties<float>::builder()
                              .with_mass(1.0f)
                              .with_molecular_mass(1.0f)
                              .build();

    const auto out = kernel.internal_energy(
        incident,
        Vector3F(1.0f, 0.0f, 0.0f),
        Vector3F(1.0f, 0.0f, 0.0f),
        material);

    EXPECT_NEAR(out.translational, incident.translational, tol);
    EXPECT_NEAR(out.rotational, incident.rotational, tol);
    EXPECT_NEAR(out.vibrational, incident.vibrational, tol);
}

TEST(SurfaceInteractionKernel, MaxwellianInteractionAppliesInternalEnergyAccommodation) {
    const auto maxwellian = MaxwellianSurfaceInteraction<float>::builder()
                                .with_accommodation(1.0f, 1.0f, 1.0f, 1.0f)
                                .build();
    const SurfaceInteractionKernel<float> kernel(maxwellian);
    const FluidInternalEnergy<float> incident { 1.0f, 2.0f, 3.0f };
    const auto material = MaterialProperties<float>::builder()
                              .with_mass(1.0f)
                              .with_molecular_mass(1.0f)
                              .with_rotational_dof(2)
                              .with_vibrational_dof(2)
                              .build();
    const auto expected = maxwellian.internal_energy(
        incident,
        Vector3F(1.0f, 0.0f, 0.0f),
        Vector3F(1.0f, 0.0f, 0.0f),
        material);

    const auto out = kernel.internal_energy(
        incident,
        Vector3F(1.0f, 0.0f, 0.0f),
        Vector3F(1.0f, 0.0f, 0.0f),
        material);

    EXPECT_NEAR(out.translational, expected.translational, tol);
    EXPECT_NEAR(out.rotational, expected.rotational, tol);
    EXPECT_NEAR(out.vibrational, expected.vibrational, tol);
}
