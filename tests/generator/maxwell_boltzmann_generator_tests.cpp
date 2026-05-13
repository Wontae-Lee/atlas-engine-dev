#include "../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>

#include <testkit/testkit.h>

namespace {

using atlas::GenerateType;
using atlas::MaxwellBoltzmannGenerateOperator;
using atlas::MaxwellBoltzmannGenerator;
using atlas::Vector3F;
using atlas::test::is_finite_vec;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(MaxwellBoltzmannGenerator, OperatorReturnsZeroForInvalidPhysicalParameters) {
    // Arrange: create a deterministic Maxwell-Boltzmann generate operator.
    const MaxwellBoltzmannGenerateOperator<float> generator(13u, Vector3F(1, 2, 3));

    // Assert: invalid physical parameters produce a zero vector.
    EXPECT_TRUE(vec_near(generator.generate(0.0f, 1.0f), Vector3F(0, 0, 0), 0.0f));
    EXPECT_TRUE(vec_near(generator.generate(300.0f, 0.0f), Vector3F(0, 0, 0), 0.0f));
}

TEST(MaxwellBoltzmannGenerator, DirectConstructorExposesConfiguredParameters) {
    // Arrange and act: construct a Maxwell-Boltzmann generator directly.
    const MaxwellBoltzmannGenerator<float> generator(300.0f, 4.65e-26f, Vector3F(1, 2, 3), 31u);

    // Assert: configured parameters and operator type are exposed.
    EXPECT_EQ(generator.type(), GenerateType::maxwell_boltzmann);
    EXPECT_NEAR(generator.param0(), 300.0f, tol);
    EXPECT_NEAR(generator.param1(), 4.65e-26f, tol);
    EXPECT_EQ(generator.generate_operator().type, GenerateType::maxwell_boltzmann);
    EXPECT_EQ(generator.make_generate_operator().type, GenerateType::maxwell_boltzmann);
    EXPECT_TRUE(is_finite_vec(generator.generate()));
}

TEST(MaxwellBoltzmannGenerator, BuilderConstructsConfiguredGenerator) {
    // Act: build a Maxwell-Boltzmann generator through the builder.
    const auto generator = MaxwellBoltzmannGenerator<float>::builder()
                               .with_temperature(350.0f)
                               .with_molecular_mass(3.0e-26f)
                               .with_bulk_velocity(Vector3F(1, 0, 0))
                               .with_seed(9u)
                               .build();

    // Assert: builder values are preserved.
    EXPECT_NEAR(generator.param0(), 350.0f, tol);
    EXPECT_NEAR(generator.param1(), 3.0e-26f, tol);
}

TEST(MaxwellBoltzmannGenerator, BuilderRejectsMissingOrInvalidParameters) {
    // Assert: missing or invalid physical parameters are rejected.
    EXPECT_THROW(
        MaxwellBoltzmannGenerator<float>::builder()
            .with_molecular_mass(1.0f)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        MaxwellBoltzmannGenerator<float>::builder()
            .with_temperature(0.0f)
            .with_molecular_mass(1.0f)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        MaxwellBoltzmannGenerator<float>::builder()
            .with_temperature(300.0f)
            .with_molecular_mass(0.0f)
            .build(),
        std::runtime_error);
}

TEST(MaxwellBoltzmannGenerator, MakeHostSharedReturnsUsableGenerator) {
    // Act: build a shared Maxwell-Boltzmann generator.
    const auto generator = MaxwellBoltzmannGenerator<float>::builder()
                               .with_temperature(300.0f)
                               .with_molecular_mass(4.65e-26f)
                               .make_host_shared();

    // Assert: the shared generator exists and has the expected type.
    ASSERT_NE(generator, nullptr);
    EXPECT_EQ(generator->type(), GenerateType::maxwell_boltzmann);
}
