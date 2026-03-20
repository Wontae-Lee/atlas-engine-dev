#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <stdexcept>

using namespace atlas;

TEST(UniformGenerator, GenerateUsesStoredParameters) {
    DeviceBuffer<Vector3<double>> expected(24);
    DeviceBuffer<Vector3<double>> actual(24);

    UniformGenerateOperator<double> {}.generate(expected, -3.0, 4.0, 21u);
    UniformGenerator<double> generator(-3.0, 4.0, 21u);
    generator.generate(actual);

    EXPECT_TRUE(test::point_buffers_near(expected, actual, eps));
}

TEST(UniformGenerator, TypeReturnsUniform) {
    UniformGenerator<double> generator(-1.0, 1.0, 1u);

    EXPECT_EQ(generator.type(), GenerateType::uniform);
}

TEST(UniformGenerator, BuilderBuildsAndValidatesRange) {
    const auto generator = UniformGenerator<double>::builder()
                               .with_min_value(-2.0)
                               .with_max_value(5.0)
                               .with_seed(8u)
                               .build();

    EXPECT_EQ(generator.type(), GenerateType::uniform);

    EXPECT_THROW(
        UniformGenerator<double>::builder()
            .with_min_value(3.0)
            .with_max_value(3.0)
            .build(),
        std::runtime_error);
}
