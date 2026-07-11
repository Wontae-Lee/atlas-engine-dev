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

TEST(UniformGenerator, GenerateWithZeroCountIsNoOp) {
    const auto         generator = make_generator();
    FluidVelocityState velocities(4);
    FluidSpeciesState  species(4);

    EXPECT_EQ(generator.generate(&velocities, &species, 0, 0), 0);
}

TEST(UniformGenerator, BuilderConfiguresTemperatureAndBulk) {
    // The temperature and bulk-velocity accessors should echo the builder inputs
    // even though this leaf does not use temperature when sampling.
    const auto generator = UniformGenerator::builder()
                               .with_species_ratios({ 1.0f })
                               .with_species_numbers({ 3.0f })
                               .with_temperature(250.0f)
                               .with_min_value(-1.0f)
                               .with_max_value(1.0f)
                               .with_bulk_velocity(Float3(1.0f, -2.0f, 3.0f))
                               .build();

    EXPECT_NEAR(generator.temperature(), 250.0f, tol);

    const Float3 bulk = generator.bulk_velocity();
    EXPECT_NEAR(bulk.x, 1.0f, tol);
    EXPECT_NEAR(bulk.y, -2.0f, tol);
    EXPECT_NEAR(bulk.z, 3.0f, tol);
}

TEST(UniformGenerator, SetBulkVelocityUpdatesGetter) {
    auto generator = make_generator();

    generator.set_bulk_velocity(Float3(4.0f, 5.0f, 6.0f));

    const Float3 bulk = generator.bulk_velocity();
    EXPECT_NEAR(bulk.x, 4.0f, tol);
    EXPECT_NEAR(bulk.y, 5.0f, tol);
    EXPECT_NEAR(bulk.z, 6.0f, tol);
}

TEST(UniformGenerator, DegenerateRangeProducesConstantComponents) {
    // min == max collapses the uniform draw, so every component equals that value
    // plus the (here zero) bulk drift, regardless of the RNG.
    const auto         generator = UniformGenerator::builder()
                               .with_species_ratios({ 1.0f })
                               .with_species_numbers({ 3.0f })
                               .with_min_value(2.0f)
                               .with_max_value(2.0f)
                               .with_seed(7u)
                               .build();
    const std::size_t  count = 32;
    FluidVelocityState velocities(count);
    FluidSpeciesState  species(count);

    ASSERT_EQ(generator.generate(&velocities, &species, 0, count), static_cast<int>(count));

    for (std::size_t i = 0; i < count; ++i) {
        const Float3 v = velocities.data()[i];
        EXPECT_NEAR(v.x, 2.0f, tol);
        EXPECT_NEAR(v.y, 2.0f, tol);
        EXPECT_NEAR(v.z, 2.0f, tol);
    }
}

TEST(UniformGenerator, SameSeedProducesIdenticalOutput) {
    // The kernel derives its RNG solely from (seed, particle index), so repeated
    // generate calls on the same leaf must be bit-for-bit reproducible.
    const auto        generator = make_generator();
    const std::size_t count     = 16;

    FluidVelocityState first_velocities(count);
    FluidSpeciesState  first_species(count);
    FluidVelocityState second_velocities(count);
    FluidSpeciesState  second_species(count);

    ASSERT_EQ(generator.generate(&first_velocities, &first_species, 0, count), static_cast<int>(count));
    ASSERT_EQ(generator.generate(&second_velocities, &second_species, 0, count), static_cast<int>(count));

    for (std::size_t i = 0; i < count; ++i) {
        const Float3 a = first_velocities.data()[i];
        const Float3 b = second_velocities.data()[i];
        EXPECT_EQ(a.x, b.x);
        EXPECT_EQ(a.y, b.y);
        EXPECT_EQ(a.z, b.z);
        EXPECT_EQ(first_species.data()[i], second_species.data()[i]);
    }
}
