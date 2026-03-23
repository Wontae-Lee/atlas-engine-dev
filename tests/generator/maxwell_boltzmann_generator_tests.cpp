#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <stdexcept>

using namespace atlas;

TEST(MaxwellBoltzmannGenerator, GenerateUsesStoredThermalParameters) {
    const Vector3<double> bulk_velocity(0.25, -0.5, 1.0);
    const auto expected = MaxwellBoltzmannGenerateOperator<double>(14u, bulk_velocity).generate(
        325.0,
        4.65e-26);

    MaxwellBoltzmannGenerator<double> generator(
        325.0,
        4.65e-26,
        bulk_velocity,
        14u);
    const auto actual = generator.generate();

    EXPECT_TRUE(test::vec_near(expected, actual, eps));
    EXPECT_EQ(generator.generate_operator().type, GenerateType::maxwell_boltzmann);
    EXPECT_TRUE(test::near(generator.param0(), 325.0, eps));
    EXPECT_TRUE(test::near(generator.param1(), 4.65e-26, eps));
}

TEST(MaxwellBoltzmannGenerator, TypeReturnsMaxwellBoltzmann) {
    MaxwellBoltzmannGenerator<double> generator(
        300.0,
        4.65e-26,
        Vector3<double>(0.0, 0.0, 0.0),
        1u);

    EXPECT_EQ(generator.type(), GenerateType::maxwell_boltzmann);
}

TEST(MaxwellBoltzmannGenerator, BuilderBuildsAndRejectsInvalidPhysicalParameters) {
    const auto generator = MaxwellBoltzmannGenerator<double>::builder()
                               .with_temperature(300.0)
                               .with_molecular_mass(4.65e-26)
                               .with_bulk_velocity(Vector3<double>(1.0, 0.0, 0.0))
                               .with_seed(4u)
                               .build();

    EXPECT_EQ(generator.type(), GenerateType::maxwell_boltzmann);

    EXPECT_THROW(
        MaxwellBoltzmannGenerator<double>::builder()
            .with_temperature(0.0)
            .with_molecular_mass(4.65e-26)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        MaxwellBoltzmannGenerator<double>::builder()
            .with_temperature(300.0)
            .build(),
        std::runtime_error);
}
