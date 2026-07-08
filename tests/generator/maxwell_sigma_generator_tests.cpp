#include <atlas/generator/maxwell_sigma_generator.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace {

using atlas::FluidSpeciesState;
using atlas::FluidTemperatureState;
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
    FluidTemperatureState temperatures(count);
    FluidSpeciesState     species(count);

    const int filled = generator.generate(&velocities, &temperatures, &species, 0, count);

    EXPECT_EQ(filled, static_cast<int>(count));
    EXPECT_EQ(species.data()[0], std::size_t { 9 });
    EXPECT_NEAR(temperatures.data()[0], 273.15f, tol);
    EXPECT_TRUE(atlas::isfinite(velocities.data()[0]));
}
