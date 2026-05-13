#include "../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>

#include <testkit/testkit.h>

namespace {

using atlas::GenerateOperator;
using atlas::GenerateType;
using atlas::MaxwellBoltzmannGenerateOperator;
using atlas::MaxwellSigmaGenerateOperator;
using atlas::UniformGenerateOperator;
using atlas::Vector3F;
using atlas::test::is_finite_vec;
using atlas::test::vec_near;

} // namespace

TEST(GenerateOperator, DefaultConstructorCreatesUniformVariant) {
    // Arrange and act: default-construct a generate operator.
    const GenerateOperator<float> generator;

    // Assert: the default variant is uniform.
    EXPECT_EQ(generator.type, GenerateType::uniform);
}

TEST(GenerateOperator, ExplicitTypeConstructorSelectsRequestedVariant) {
    // Arrange and act: construct explicit generate operator variants.
    const GenerateOperator<float> sigma(GenerateType::maxwell_sigma, 11u);
    const GenerateOperator<float> boltzmann(GenerateType::maxwell_boltzmann, 13u);

    // Assert: the requested variants are selected.
    EXPECT_EQ(sigma.type, GenerateType::maxwell_sigma);
    EXPECT_EQ(boltzmann.type, GenerateType::maxwell_boltzmann);
}

TEST(GenerateOperator, CopyConstructionAndAssignmentPreserveType) {
    // Arrange: create a non-default operator variant.
    const GenerateOperator<float> original(GenerateType::maxwell_sigma, 17u);

    // Act: copy-construct and copy-assign the operator.
    const GenerateOperator<float> copied(original);
    GenerateOperator<float> assigned;
    assigned = original;

    // Assert: both copies preserve the variant type.
    EXPECT_EQ(copied.type, GenerateType::maxwell_sigma);
    EXPECT_EQ(assigned.type, GenerateType::maxwell_sigma);
}

TEST(GenerateOperator, ConcreteOperatorConstructorsWrapCorrectVariant) {
    // Arrange and act: wrap concrete generate operator implementations.
    const GenerateOperator<float> uniform(UniformGenerateOperator<float>(1u));
    const GenerateOperator<float> sigma(MaxwellSigmaGenerateOperator<float>(2u));
    const GenerateOperator<float> boltzmann(MaxwellBoltzmannGenerateOperator<float>(3u));

    // Assert: wrapper variants match the concrete operator kind.
    EXPECT_EQ(uniform.type, GenerateType::uniform);
    EXPECT_EQ(sigma.type, GenerateType::maxwell_sigma);
    EXPECT_EQ(boltzmann.type, GenerateType::maxwell_boltzmann);
}

TEST(GenerateOperator, GenerateReturnsFiniteValuesForSupportedVariants) {
    // Arrange: create each supported generate operator variant.
    const GenerateOperator<float> uniform(GenerateType::uniform, 5u);
    const GenerateOperator<float> sigma(GenerateType::maxwell_sigma, 7u);
    const GenerateOperator<float> boltzmann(GenerateType::maxwell_boltzmann, 9u);

    // Assert: all variants produce finite vectors for valid inputs.
    EXPECT_TRUE(is_finite_vec(uniform.generate(-1.0f, 1.0f)));
    EXPECT_TRUE(is_finite_vec(sigma.generate(0.5f, 0.0f)));
    EXPECT_TRUE(is_finite_vec(boltzmann.generate(300.0f, 1.0e-26f)));
}

TEST(GenerateOperator, InvalidPhysicalParametersReturnZeroForBoltzmannAndSigma) {
    // Arrange: create physical-model generate operator variants.
    const GenerateOperator<float> sigma(GenerateType::maxwell_sigma, 7u);
    const GenerateOperator<float> boltzmann(GenerateType::maxwell_boltzmann, 9u);

    // Assert: invalid physical parameters produce zero vectors.
    EXPECT_TRUE(vec_near(sigma.generate(0.0f, 0.0f), Vector3F(0, 0, 0), 0.0f));
    EXPECT_TRUE(vec_near(boltzmann.generate(0.0f, 1.0f), Vector3F(0, 0, 0), 0.0f));
    EXPECT_TRUE(vec_near(boltzmann.generate(300.0f, 0.0f), Vector3F(0, 0, 0), 0.0f));
}
