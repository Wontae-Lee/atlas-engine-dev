#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/sparta/sparta_vss_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::SpartaVssKernel;

MaterialProperties<float>
make_properties() {
    return MaterialProperties<float>::builder()
        .with_type(MaterialType::Molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_reference_diameter(1.0f)
        .with_reference_temperature(1.0f)
        .with_viscosity_index(0.75f)
        .with_scattering_parameter(1.2f)
        .build();
}

} // namespace

TEST(SpartaVssKernel, CollisionFrequencyIsPositiveForPositiveRelativeSpeed) {
    const auto properties = make_properties();

    EXPECT_GT(SpartaVssKernel<float>::collision_frequency(properties, properties, 4.0f), 0.0f);
}

TEST(SpartaVssKernel, CollisionFrequencyUsesSpartaRelativeSpeedSquaredExponent) {
    const auto properties = make_properties();
    const float slow = SpartaVssKernel<float>::collision_frequency(properties, properties, 1.0f);
    const float fast = SpartaVssKernel<float>::collision_frequency(properties, properties, 4.0f);

    EXPECT_GT(fast, slow);
}
