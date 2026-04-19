#include "../utilities/tests_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(MaxwellBoltzmannGenerator, OperatorReturnsZeroForInvalidPhysicalParameters) {
    const atlas::fluid::MaxwellBoltzmannGenerateOperator<T> generator(13u, Vec3(1, 2, 3));

    EXPECT_TRUE(atlas::test::vec_near(generator.generate(0.0f, 1.0f), Vec3(0, 0, 0), 0.0f));
    EXPECT_TRUE(atlas::test::vec_near(generator.generate(300.0f, 0.0f), Vec3(0, 0, 0), 0.0f));
}

TEST(MaxwellBoltzmannGenerator, DirectConstructorExposesConfiguredParameters) {
    const atlas::fluid::MaxwellBoltzmannGenerator<T> generator(300.0f, 4.65e-26f, Vec3(1, 2, 3), 31u);

    EXPECT_EQ(generator.type(), atlas::fluid::GenerateType::maxwell_boltzmann);
    EXPECT_NEAR(generator.param0(), 300.0f, kEps);
    EXPECT_NEAR(generator.param1(), 4.65e-26f, 1e-30f);
    EXPECT_EQ(generator.generate_operator().type, atlas::fluid::GenerateType::maxwell_boltzmann);
    EXPECT_EQ(generator.make_generate_operator().type, atlas::fluid::GenerateType::maxwell_boltzmann);
    EXPECT_TRUE(atlas::test::is_finite_vec(generator.generate()));
}

TEST(MaxwellBoltzmannGenerator, BuilderConstructsConfiguredGenerator) {
    const auto generator = atlas::fluid::MaxwellBoltzmannGenerator<T>::builder()
                               .with_temperature(350.0f)
                               .with_molecular_mass(3.0e-26f)
                               .with_bulk_velocity(Vec3(1, 0, 0))
                               .with_seed(9u)
                               .build();

    EXPECT_NEAR(generator.param0(), 350.0f, kEps);
    EXPECT_NEAR(generator.param1(), 3.0e-26f, 1e-30f);
}

TEST(MaxwellBoltzmannGenerator, BuilderRejectsMissingOrInvalidParameters) {
    EXPECT_THROW(
        atlas::fluid::MaxwellBoltzmannGenerator<T>::builder()
            .with_molecular_mass(1.0f)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::MaxwellBoltzmannGenerator<T>::builder()
            .with_temperature(0.0f)
            .with_molecular_mass(1.0f)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::MaxwellBoltzmannGenerator<T>::builder()
            .with_temperature(300.0f)
            .with_molecular_mass(0.0f)
            .build(),
        std::runtime_error);
}

TEST(MaxwellBoltzmannGenerator, MakeHostSharedReturnsUsableGenerator) {
    const auto generator = atlas::fluid::MaxwellBoltzmannGenerator<T>::builder()
                               .with_temperature(300.0f)
                               .with_molecular_mass(4.65e-26f)
                               .make_host_shared();

    ASSERT_NE(generator, nullptr);
    EXPECT_EQ(generator->type(), atlas::fluid::GenerateType::maxwell_boltzmann);
}
