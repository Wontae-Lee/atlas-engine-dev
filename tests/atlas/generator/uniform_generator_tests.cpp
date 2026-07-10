#include <atlas/generator/uniform_generator.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace {

using atlas::FluidSpeciesState;
using atlas::FluidVelocityState;
using atlas::UniformGenerator;
using atlas::Float3;
using atlas::tol;

UniformGenerator
make_generator() {
    return UniformGenerator::builder()
        .with_species_ratios({ 1.0f })
        .with_species_numbers({ 3.0f })
        .with_temperature(250.0f)
        .with_min_value(-2.0f)
        .with_max_value(3.0f)
        .with_seed(7u)
        .build();
}

}

TEST(UniformGenerator, BuilderRejectsEmptySpecies) {
    EXPECT_THROW(
        static_cast<void>(UniformGenerator::builder().with_min_value(0.0f).with_max_value(1.0f).build()),
        std::runtime_error);
}

TEST(UniformGenerator, BuilderRejectsSpeciesSizeMismatch) {
    EXPECT_THROW(
        static_cast<void>(UniformGenerator::builder().with_species_ratios({ 0.5f, 0.5f }).with_species_numbers({ 3.0f }).build()),
        std::runtime_error);
}

TEST(UniformGenerator, BuilderRejectsInvertedRange) {
    EXPECT_THROW(
        static_cast<void>(UniformGenerator::builder().with_species_ratios({ 1.0f }).with_species_numbers({ 3.0f }).with_min_value(2.0f).with_max_value(1.0f).build()),
        std::runtime_error);
}

TEST(UniformGenerator, GenerateFillsStatesAndReturnsCount) {
    const auto            generator = make_generator();
    const std::size_t     count     = 8;
    FluidVelocityState    velocities(count);
    FluidSpeciesState     species(count);

    const int filled = generator.generate(&velocities, &species, 0, count);

    EXPECT_EQ(filled, static_cast<int>(count));
    EXPECT_EQ(species.data()[0], std::size_t { 3 });

    const Float3 v = velocities.data()[0];
    EXPECT_TRUE(atlas::isfinite(v));
    EXPECT_GE(v.x, -2.0f);
    EXPECT_LE(v.x, 3.0f);
}

TEST(UniformGenerator, GenerateRejectsNullState) {
    const auto            generator = make_generator();
    FluidSpeciesState     species(4);

    EXPECT_EQ(generator.generate(nullptr, &species, 0, 4), 0);
}
