#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/piclas/piclas_vhs_kernel.h>

#include <testkit/testkit.h>

namespace {

using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::PiclasVhsKernel;

MaterialProperties<float>
make_properties() {
    return MaterialProperties<float>::builder()
        .with_type(MaterialType::Molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_reference_diameter(1.0f)
        .with_reference_temperature(1.0f)
        .with_viscosity_index(0.75f)
        .build();
}

} // namespace

TEST(PiclasVhsKernel, SigmaGIsPositiveForPositiveRelativeSpeed) {
    const auto properties = make_properties();

    EXPECT_GT(PiclasVhsKernel<float>::sigma_g(properties, properties, 4.0f), 0.0f);
}

TEST(PiclasVhsKernel, CollisionProbabilityUsesPairCaseCount) {
    const float probability = PiclasVhsKernel<float>::collision_probability(
        2.0f,
        4.0f,
        3.0f,
        false,
        2,
        5.0f,
        0.25f,
        10.0f);

    EXPECT_FLOAT_EQ(probability, 1.5f);
}
