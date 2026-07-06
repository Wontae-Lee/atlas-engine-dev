#include <atlas/generator/generate.h>

#include <gtest/gtest.h>

namespace {

using atlas::Generate;
using atlas::GenerateType;
using atlas::MaxwellBoltzmannGenerate;
using atlas::MaxwellSigmaGenerate;
using atlas::UniformGenerate;
using atlas::Float3;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, atlas::tol);
    EXPECT_NEAR(actual.y, expected.y, atlas::tol);
    EXPECT_NEAR(actual.z, expected.z, atlas::tol);
}

}

TEST(Generate, DefaultConstructorCreatesUniformVariant) {
    const Generate generator;

    EXPECT_EQ(generator.type, GenerateType::uniform);
}

TEST(Generate, ExplicitTypeConstructorSelectsRequestedVariant) {
    const Generate sigma(GenerateType::maxwell_sigma, 11u);
    const Generate boltzmann(GenerateType::maxwell_boltzmann, 13u);

    EXPECT_EQ(sigma.type, GenerateType::maxwell_sigma);
    EXPECT_EQ(boltzmann.type, GenerateType::maxwell_boltzmann);
}

TEST(Generate, CopyConstructionAndAssignmentPreserveType) {
    const Generate original(GenerateType::maxwell_sigma, 17u);

    const Generate copied(original);
    Generate assigned;
    assigned = original;

    EXPECT_EQ(copied.type, GenerateType::maxwell_sigma);
    EXPECT_EQ(assigned.type, GenerateType::maxwell_sigma);
}

TEST(Generate, ConcreteOperatorConstructorsWrapCorrectVariant) {
    const Generate uniform(UniformGenerate(1u));
    const Generate sigma(MaxwellSigmaGenerate(2u));
    const Generate boltzmann(MaxwellBoltzmannGenerate(3u));

    EXPECT_EQ(uniform.type, GenerateType::uniform);
    EXPECT_EQ(sigma.type, GenerateType::maxwell_sigma);
    EXPECT_EQ(boltzmann.type, GenerateType::maxwell_boltzmann);
}

TEST(Generate, GenerateReturnsFiniteValuesForSupportedVariants) {
    const Generate uniform(GenerateType::uniform, 5u);
    const Generate sigma(GenerateType::maxwell_sigma, 7u);
    const Generate boltzmann(GenerateType::maxwell_boltzmann, 9u);

    EXPECT_TRUE(atlas::isfinite(uniform.generate(-1.0f, 1.0f)));
    EXPECT_TRUE(atlas::isfinite(sigma.generate(0.5f, 0.0f)));
    EXPECT_TRUE(atlas::isfinite(boltzmann.generate(300.0f, 1.0e-26f)));
}

TEST(Generate, InvalidPhysicalParametersReturnZeroForBoltzmannAndSigma) {
    const Generate sigma(GenerateType::maxwell_sigma, 7u);
    const Generate boltzmann(GenerateType::maxwell_boltzmann, 9u);

    expect_vec_near(sigma.generate(0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(boltzmann.generate(0.0f, 1.0f), Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(boltzmann.generate(300.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f));
}
