#include "../../utilities/tests_utils.h"

#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>

#include <gtest/gtest.h>

namespace {

using T = float;

atlas::MatrialProperties<T>
make_properties() {
    return atlas::MatrialProperties<T>::builder()
        .with_type(atlas::MaterialType::Molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_collision_diameter(1.0f)
        .with_viscosity_index(0.75f)
        .with_scattering_parameter(1.25f)
        .build();
}

} // namespace

TEST(DsmcKernel, DefaultConstructorSelectsHardSphere) {
    const atlas::system::DsmcKernel<T> kernel;

    EXPECT_EQ(kernel.type, atlas::system::DsmcKernelType::hard_sphere);
}

TEST(DsmcKernel, CrossSectionDispatchesForAllKernelTypes) {
    const auto properties = make_properties();

    EXPECT_GT(atlas::system::DsmcKernel<T>::cross_section(
                  atlas::system::DsmcKernelType::hard_sphere, properties, properties, 2.0f),
              0.0f);
    EXPECT_GT(atlas::system::DsmcKernel<T>::cross_section(
                  atlas::system::DsmcKernelType::variable_hard_sphere, properties, properties, 2.0f),
              0.0f);
    EXPECT_GT(atlas::system::DsmcKernel<T>::cross_section(
                  atlas::system::DsmcKernelType::variable_soft_sphere, properties, properties, 2.0f),
              0.0f);
}
