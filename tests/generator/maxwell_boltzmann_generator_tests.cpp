#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <stdexcept>

using namespace atlas;

TEST(MaxwellBoltzmannGenerator, GenerateUsesStoredThermalParameters) {
    DeviceBuffer<Vector3<double>> expected(24);
    DeviceBuffer<Vector3<double>> actual(24);
    const Vector3<double> bulk_velocity(0.25, -0.5, 1.0);

    MaxwellBoltzmannGenerateOperator<double> {}.generate(
        expected,
        325.0,
        4.65e-26,
        bulk_velocity,
        14u);

    MaxwellBoltzmannGenerator<double> generator(
        325.0,
        4.65e-26,
        bulk_velocity,
        14u);
    generator.generate(actual);

    EXPECT_TRUE(test::point_buffers_near(expected, actual, eps));
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
