#include "../../utilities/test_utils.h"

#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::system::DsmcKernel;
using atlas::system::DsmcKernelType;

MaterialProperties<float>
make_properties() {
    return MaterialProperties<float>::builder()
        .with_type(MaterialType::Molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_reference_diameter(1.0f)
        .with_reference_temperature(1.0f)
        .with_viscosity_index(0.75f)
        .with_scattering_parameter(1.25f)
        .build();
}

} // namespace

TEST(DsmcKernel, DefaultConstructorSelectsHardSphere) {
    // Arrange: create a default runtime DSMC kernel.
    const DsmcKernel<float> kernel;

    // Assert: the default type is hard sphere.
    EXPECT_EQ(kernel.type, DsmcKernelType::hard_sphere);
}

TEST(DsmcKernel, CrossSectionDispatchesForAllKernelTypes) {
    // Arrange: create representative material properties.
    const auto properties = make_properties();

    // Assert: every runtime kernel variant reports a positive cross section.
    EXPECT_GT(DsmcKernel<float>::cross_section(
                  DsmcKernelType::hard_sphere, properties, properties, 2.0f),
              0.0f);
    EXPECT_GT(DsmcKernel<float>::cross_section(
                  DsmcKernelType::variable_hard_sphere, properties, properties, 2.0f),
              0.0f);
    EXPECT_GT(DsmcKernel<float>::cross_section(
                  DsmcKernelType::variable_soft_sphere, properties, properties, 2.0f),
              0.0f);
}

TEST(DsmcKernel, PairParametersUseSpeciesPairAverages) {
    const auto lhs = MaterialProperties<float>::builder()
                         .with_type(MaterialType::Molecule)
                         .with_mass(1.0f)
                         .with_molecular_mass(2.0f)
                         .with_reference_diameter(2.0f)
                         .with_reference_temperature(100.0f)
                         .with_viscosity_index(0.6f)
                         .with_scattering_parameter(1.0f)
                         .build();
    const auto rhs = MaterialProperties<float>::builder()
                         .with_type(MaterialType::Molecule)
                         .with_mass(1.0f)
                         .with_molecular_mass(6.0f)
                         .with_reference_diameter(4.0f)
                         .with_reference_temperature(300.0f)
                         .with_viscosity_index(0.8f)
                         .with_scattering_parameter(2.0f)
                         .build();

    const auto pair = DsmcKernel<float>::pair_parameters(lhs, rhs);

    EXPECT_TRUE(pair.valid);
    EXPECT_NEAR(pair.reference_diameter, 3.0f, 1.0e-6f);
    EXPECT_NEAR(pair.reference_temperature, 200.0f, 1.0e-6f);
    EXPECT_NEAR(pair.viscosity_index, 0.7f, 1.0e-6f);
    EXPECT_NEAR(pair.scattering_parameter, 1.5f, 1.0e-6f);
    EXPECT_NEAR(pair.reduced_mass, 1.5f, 1.0e-6f);
}
