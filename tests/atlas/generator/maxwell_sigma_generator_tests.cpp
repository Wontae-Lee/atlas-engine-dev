#include <atlas/generator/maxwell_sigma_generator.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace {

using atlas::FluidSpeciesState;
using atlas::FluidVelocityState;
using atlas::MaxwellSigmaGenerator;
using atlas::Float3;
using atlas::tol;

MaxwellSigmaGenerator
make_generator() {
    return MaxwellSigmaGenerator::builder()
        .with_species_ratios({ 1.0f })
        .with_species_numbers({ 9.0f })
        .with_temperature(273.15f)
        .with_sigma(2.0f)
        .with_seed(5u)
        .build();
}

}

TEST(MaxwellSigmaGenerator, BuilderRejectsEmptySpecies) {
    EXPECT_THROW(
        static_cast<void>(MaxwellSigmaGenerator::builder().with_sigma(1.0f).build()),
        std::runtime_error);
}

TEST(MaxwellSigmaGenerator, BuilderRejectsNegativeSigma) {
    EXPECT_THROW(
        static_cast<void>(MaxwellSigmaGenerator::builder().with_species_ratios({ 1.0f }).with_species_numbers({ 9.0f }).with_sigma(-1.0f).build()),
        std::runtime_error);
}

TEST(MaxwellSigmaGenerator, GenerateFillsStatesAndReturnsCount) {
    const auto            generator = make_generator();
    const std::size_t     count     = 8;
    FluidVelocityState    velocities(count);
    FluidSpeciesState     species(count);

    const int filled = generator.generate(&velocities, &species, 0, count);

    EXPECT_EQ(filled, static_cast<int>(count));
    EXPECT_EQ(species.data()[0], std::size_t { 9 });
    EXPECT_TRUE(atlas::isfinite(velocities.data()[0]));
}

TEST(MaxwellSigmaGenerator, GenerateRejectsNullState) {
    const auto        generator = make_generator();
    FluidSpeciesState species(4);

    EXPECT_EQ(generator.generate(nullptr, &species, 0, 4), 0);
}

TEST(MaxwellSigmaGenerator, GenerateWithZeroCountIsNoOp) {
    const auto         generator = make_generator();
    FluidVelocityState velocities(4);
    FluidSpeciesState  species(4);

    EXPECT_EQ(generator.generate(&velocities, &species, 0, 0), 0);
}

TEST(MaxwellSigmaGenerator, ComponentsAreFiniteAndRoughlyZeroMean) {
    // sigma 2, no bulk drift: each component is a zero-mean Gaussian, so the
    // per-component sample mean sits near zero. The band is far wider than the
    // sqrt(N) standard error, so a fixed seed cannot flake it.
    const auto         generator = make_generator();
    const std::size_t  count     = 4096;
    FluidVelocityState velocities(count);
    FluidSpeciesState  species(count);

    ASSERT_EQ(generator.generate(&velocities, &species, 0, count), static_cast<int>(count));

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const Float3 v = velocities.data()[i];
        ASSERT_TRUE(atlas::isfinite(v));
        sum_x += v.x;
        sum_y += v.y;
        sum_z += v.z;
    }

    const double n = static_cast<double>(count);
    EXPECT_NEAR(sum_x / n, 0.0, 0.5);
    EXPECT_NEAR(sum_y / n, 0.0, 0.5);
    EXPECT_NEAR(sum_z / n, 0.0, 0.5);
}

TEST(MaxwellSigmaGenerator, ZeroSigmaProducesBulkDriftOnly) {
    // With sigma == 0 the Gaussian collapses, so every particle receives exactly
    // the bulk drift and nothing else.
    const auto         generator = MaxwellSigmaGenerator::builder()
                               .with_species_ratios({ 1.0f })
                               .with_species_numbers({ 9.0f })
                               .with_sigma(0.0f)
                               .with_bulk_velocity(Float3(1.0f, 2.0f, 3.0f))
                               .with_seed(5u)
                               .build();
    const std::size_t  count = 32;
    FluidVelocityState velocities(count);
    FluidSpeciesState  species(count);

    ASSERT_EQ(generator.generate(&velocities, &species, 0, count), static_cast<int>(count));

    for (std::size_t i = 0; i < count; ++i) {
        const Float3 v = velocities.data()[i];
        EXPECT_NEAR(v.x, 1.0f, tol);
        EXPECT_NEAR(v.y, 2.0f, tol);
        EXPECT_NEAR(v.z, 3.0f, tol);
    }
}

TEST(MaxwellSigmaGenerator, SetBulkVelocityUpdatesGetter) {
    auto generator = make_generator();

    generator.set_bulk_velocity(Float3(-1.0f, 0.5f, 2.0f));

    const Float3 bulk = generator.bulk_velocity();
    EXPECT_NEAR(bulk.x, -1.0f, tol);
    EXPECT_NEAR(bulk.y, 0.5f, tol);
    EXPECT_NEAR(bulk.z, 2.0f, tol);
}
