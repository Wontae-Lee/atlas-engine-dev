#include <atlas/generator/maxwell_boltzmann_generator.h>

#include <atlas/generator/generate.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using atlas::GenerateType;
using atlas::MaxwellBoltzmannGenerate;
using atlas::MaxwellBoltzmannGenerator;
using atlas::Float3;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, atlas::tol);
    EXPECT_NEAR(actual.y, expected.y, atlas::tol);
    EXPECT_NEAR(actual.z, expected.z, atlas::tol);
}

}

TEST(MaxwellBoltzmannGenerator, OperatorReturnsZeroForInvalidPhysicalParameters) {
    const MaxwellBoltzmannGenerate generator(13u, Float3(1.0f, 2.0f, 3.0f));

    expect_vec_near(generator.generate(0.0f, 1.0f), Float3(0.0f, 0.0f, 0.0f));
    expect_vec_near(generator.generate(300.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f));
}

TEST(MaxwellBoltzmannGenerator, DirectConstructorExposesConfiguredParameters) {
    const MaxwellBoltzmannGenerator generator(300.0f, 4.65e-26f, Float3(1.0f, 2.0f, 3.0f), 31u);

    EXPECT_EQ(generator.type(), GenerateType::maxwell_boltzmann);
    EXPECT_NEAR(generator.param0(), 300.0f, atlas::tol);
    EXPECT_NEAR(generator.param1(), 4.65e-26f, atlas::tol);
    EXPECT_EQ(generator.generate_operator().type, GenerateType::maxwell_boltzmann);
    EXPECT_EQ(generator.make_generate_operator().type, GenerateType::maxwell_boltzmann);
    EXPECT_TRUE(atlas::isfinite(generator.generate()));
}

TEST(MaxwellBoltzmannGenerator, BuilderConstructsConfiguredGenerator) {
    const auto generator = MaxwellBoltzmannGenerator::builder()
                               .with_temperature(350.0f)
                               .with_molecular_mass(3.0e-26f)
                               .with_bulk_velocity(Float3(1.0f, 0.0f, 0.0f))
                               .with_seed(9u)
                               .build();

    EXPECT_NEAR(generator.param0(), 350.0f, atlas::tol);
    EXPECT_NEAR(generator.param1(), 3.0e-26f, atlas::tol);
}

TEST(MaxwellBoltzmannGenerator, BuilderRejectsMissingOrInvalidParameters) {
    EXPECT_THROW(
        static_cast<void>(MaxwellBoltzmannGenerator::builder()
                              .with_molecular_mass(1.0f)
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(MaxwellBoltzmannGenerator::builder()
                              .with_temperature(0.0f)
                              .with_molecular_mass(1.0f)
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(MaxwellBoltzmannGenerator::builder()
                              .with_temperature(300.0f)
                              .with_molecular_mass(0.0f)
                              .build()),
        std::runtime_error);
}

TEST(MaxwellBoltzmannGenerator, MakeHostSharedReturnsUsableGenerator) {
    const auto generator = MaxwellBoltzmannGenerator::builder()
                               .with_temperature(300.0f)
                               .with_molecular_mass(4.65e-26f)
                               .make_host_shared();

    ASSERT_NE(generator, nullptr);
    EXPECT_EQ(generator->type(), GenerateType::maxwell_boltzmann);
}
