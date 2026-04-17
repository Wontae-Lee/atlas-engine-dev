#include "../utilities/tests_utils.h"

#include <atlas/generator/generate_operator.h>

#include <gtest/gtest.h>

namespace {

using T = float;

} // namespace

TEST(GenerateOperator, DefaultConstructorCreatesUniformVariant) {
    const atlas::fluid::GenerateOperator<T> generator;

    EXPECT_EQ(generator.type, atlas::fluid::GenerateType::uniform);
}

TEST(GenerateOperator, ExplicitTypeConstructorSelectsRequestedVariant) {
    const atlas::fluid::GenerateOperator<T> sigma(atlas::fluid::GenerateType::maxwell_sigma, 11u);
    const atlas::fluid::GenerateOperator<T> boltzmann(atlas::fluid::GenerateType::maxwell_boltzmann, 13u);

    EXPECT_EQ(sigma.type, atlas::fluid::GenerateType::maxwell_sigma);
    EXPECT_EQ(boltzmann.type, atlas::fluid::GenerateType::maxwell_boltzmann);
}

TEST(GenerateOperator, CopyConstructionAndAssignmentPreserveType) {
    const atlas::fluid::GenerateOperator<T> original(atlas::fluid::GenerateType::maxwell_sigma, 17u);
    const atlas::fluid::GenerateOperator<T> copied(original);

    atlas::fluid::GenerateOperator<T> assigned;
    assigned = original;

    EXPECT_EQ(copied.type, atlas::fluid::GenerateType::maxwell_sigma);
    EXPECT_EQ(assigned.type, atlas::fluid::GenerateType::maxwell_sigma);
}

TEST(GenerateOperator, ConcreteOperatorConstructorsWrapCorrectVariant) {
    const atlas::fluid::GenerateOperator<T> uniform(atlas::fluid::UniformGenerateOperator<T>(1u));
    const atlas::fluid::GenerateOperator<T> sigma(atlas::fluid::MaxwellSigmaGenerateOperator<T>(2u));
    const atlas::fluid::GenerateOperator<T> boltzmann(atlas::fluid::MaxwellBoltzmannGenerateOperator<T>(3u));

    EXPECT_EQ(uniform.type, atlas::fluid::GenerateType::uniform);
    EXPECT_EQ(sigma.type, atlas::fluid::GenerateType::maxwell_sigma);
    EXPECT_EQ(boltzmann.type, atlas::fluid::GenerateType::maxwell_boltzmann);
}

TEST(GenerateOperator, GenerateReturnsFiniteValuesForSupportedVariants) {
    const atlas::fluid::GenerateOperator<T> uniform(atlas::fluid::GenerateType::uniform, 5u);
    const atlas::fluid::GenerateOperator<T> sigma(atlas::fluid::GenerateType::maxwell_sigma, 7u);
    const atlas::fluid::GenerateOperator<T> boltzmann(atlas::fluid::GenerateType::maxwell_boltzmann, 9u);

    EXPECT_TRUE(atlas::test::is_finite_vec(uniform.generate(-1.0f, 1.0f)));
    EXPECT_TRUE(atlas::test::is_finite_vec(sigma.generate(0.5f, 0.0f)));
    EXPECT_TRUE(atlas::test::is_finite_vec(boltzmann.generate(300.0f, 1.0e-26f)));
}

TEST(GenerateOperator, InvalidPhysicalParametersReturnZeroForBoltzmannAndSigma) {
    const atlas::fluid::GenerateOperator<T> sigma(atlas::fluid::GenerateType::maxwell_sigma, 7u);
    const atlas::fluid::GenerateOperator<T> boltzmann(atlas::fluid::GenerateType::maxwell_boltzmann, 9u);

    EXPECT_TRUE(atlas::test::vec_near(sigma.generate(0.0f, 0.0f), atlas::Vector3<T>(0, 0, 0), 0.0f));
    EXPECT_TRUE(atlas::test::vec_near(boltzmann.generate(0.0f, 1.0f), atlas::Vector3<T>(0, 0, 0), 0.0f));
    EXPECT_TRUE(atlas::test::vec_near(boltzmann.generate(300.0f, 0.0f), atlas::Vector3<T>(0, 0, 0), 0.0f));
}
