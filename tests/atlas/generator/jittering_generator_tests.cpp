#include <atlas/generator/jittering_generator.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace {

using atlas::FluidSpeciesState;
using atlas::FluidVelocityState;
using atlas::JitteringGenerator;
using atlas::Float3;
using atlas::tol;

JitteringGenerator
make_generator() {
    return JitteringGenerator::builder()
        .with_species_ratios({ 0.25f, 0.75f })
        .with_species_numbers({ 1.0f, 2.0f })
        .with_temperature(200.0f)
        .with_base_value(5.0f)
        .with_jitter_radius(0.5f)
        .with_seed(11u)
        .build();
}

}

TEST(JitteringGenerator, BuilderRejectsEmptySpecies) {
    EXPECT_THROW(
        static_cast<void>(JitteringGenerator::builder().with_base_value(1.0f).build()),
        std::runtime_error);
}

TEST(JitteringGenerator, BuilderRejectsSpeciesSizeMismatch) {
    EXPECT_THROW(
        static_cast<void>(JitteringGenerator::builder().with_species_ratios({ 1.0f }).with_species_numbers({ 1.0f, 2.0f }).build()),
        std::runtime_error);
}

TEST(JitteringGenerator, GenerateFillsStatesAndReturnsCount) {
    const auto            generator = make_generator();
    const std::size_t     count     = 8;
    FluidVelocityState    velocities(count);
    FluidSpeciesState     species(count);

    const int filled = generator.generate(&velocities, &species, 0, count);

    EXPECT_EQ(filled, static_cast<int>(count));

    const std::size_t sampled = species.data()[0];
    EXPECT_TRUE(sampled == std::size_t { 1 } || sampled == std::size_t { 2 });

    const Float3 v = velocities.data()[0];
    EXPECT_TRUE(atlas::isfinite(v));
    EXPECT_GE(v.x, 5.0f - 0.5f - tol);
    EXPECT_LE(v.x, 5.0f + 0.5f + tol);
}

TEST(JitteringGenerator, GenerateWritesAtOffset) {
    const auto            generator = make_generator();
    FluidVelocityState    velocities(8);
    FluidSpeciesState     species(8);

    const int filled = generator.generate(&velocities, &species, 4, 8);

    EXPECT_EQ(filled, 4);
}

TEST(JitteringGenerator, GenerateRejectsNullState) {
    const auto        generator = make_generator();
    FluidSpeciesState species(4);

    EXPECT_EQ(generator.generate(nullptr, &species, 0, 4), 0);
}

TEST(JitteringGenerator, GenerateWithZeroCountIsNoOp) {
    const auto         generator = make_generator();
    FluidVelocityState velocities(4);
    FluidSpeciesState  species(4);

    EXPECT_EQ(generator.generate(&velocities, &species, 0, 0), 0);
}

TEST(JitteringGenerator, KeepsEverySampleInsideBounds) {
    // base_value 5, jitter_radius 0.5, no bulk drift: every component of every
    // sampled velocity must land in [base - radius, base + radius].
    const auto         generator = make_generator();
    const std::size_t  count     = 256;
    FluidVelocityState velocities(count);
    FluidSpeciesState  species(count);

    ASSERT_EQ(generator.generate(&velocities, &species, 0, count), static_cast<int>(count));

    for (std::size_t i = 0; i < count; ++i) {
        const Float3 v = velocities.data()[i];
        EXPECT_TRUE(atlas::isfinite(v));
        EXPECT_GE(v.x, 5.0f - 0.5f - tol);
        EXPECT_LE(v.x, 5.0f + 0.5f + tol);
        EXPECT_GE(v.y, 5.0f - 0.5f - tol);
        EXPECT_LE(v.y, 5.0f + 0.5f + tol);
        EXPECT_GE(v.z, 5.0f - 0.5f - tol);
        EXPECT_LE(v.z, 5.0f + 0.5f + tol);
    }
}
