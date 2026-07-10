#include <atlas/generator/generator.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator_type.h>
#include <atlas/generator/jittering_generator.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/generator/maxwell_sigma_generator.h>
#include <atlas/generator/uniform_generator.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <utility>

namespace {

using atlas::FluidSpeciesState;
using atlas::FluidVelocityState;
using atlas::Generator;
using atlas::GeneratorType;
using atlas::JitteringGenerator;
using atlas::MaxwellBoltzmannGenerator;
using atlas::MaxwellSigmaGenerator;
using atlas::UniformGenerator;
using atlas::tol;

UniformGenerator
make_uniform() {
    return UniformGenerator::builder()
        .with_species_ratios({ 1.0f })
        .with_species_numbers({ 3.0f })
        .with_temperature(250.0f)
        .with_min_value(-1.0f)
        .with_max_value(1.0f)
        .with_seed(7u)
        .build();
}

MaxwellSigmaGenerator
make_maxwell_sigma() {
    return MaxwellSigmaGenerator::builder()
        .with_species_ratios({ 1.0f })
        .with_species_numbers({ 9.0f })
        .with_temperature(273.15f)
        .with_sigma(1.0f)
        .with_seed(5u)
        .build();
}

JitteringGenerator
make_jittering() {
    return JitteringGenerator::builder()
        .with_species_ratios({ 1.0f })
        .with_species_numbers({ 2.0f })
        .with_base_value(0.0f)
        .with_jitter_radius(0.5f)
        .with_seed(3u)
        .build();
}

MaxwellBoltzmannGenerator
make_maxwell_boltzmann() {
    return MaxwellBoltzmannGenerator::builder()
        .with_species_ratios({ 1.0f })
        .with_species_numbers({ 4.0f })
        .with_species_mass({ 2.0f })
        .with_temperature(300.0f)
        .with_seed(6u)
        .build();
}

}

TEST(Generator, DefaultConstructsUniform) {
    const Generator generator {};

    EXPECT_EQ(generator.type, GeneratorType::uniform);
}

TEST(Generator, WrapsMaxwellSigmaLeaf) {
    const Generator generator(make_maxwell_sigma());

    EXPECT_EQ(generator.type, GeneratorType::maxwell_sigma);
}

TEST(Generator, GenerateDispatchesToLeaf) {
    const Generator       generator(make_uniform());
    const std::size_t     count = 8;
    FluidVelocityState    velocities(count);
    FluidSpeciesState     species(count);

    const int filled = generator.generate(&velocities, &species, 0, count);

    EXPECT_EQ(filled, static_cast<int>(count));
    EXPECT_EQ(species.data()[0], std::size_t { 3 });
}

TEST(Generator, MoveConstructPreservesBehaviour) {
    Generator       source(make_maxwell_sigma());
    const Generator moved = std::move(source);

    EXPECT_EQ(moved.type, GeneratorType::maxwell_sigma);

    FluidVelocityState    velocities(4);
    FluidSpeciesState     species(4);
    EXPECT_EQ(moved.generate(&velocities, &species, 0, 4), 4);
}

TEST(Generator, WrapsJitteringLeaf) {
    const Generator generator(make_jittering());

    EXPECT_EQ(generator.type, GeneratorType::jittering);
}

TEST(Generator, WrapsMaxwellBoltzmannLeaf) {
    const Generator generator(make_maxwell_boltzmann());

    EXPECT_EQ(generator.type, GeneratorType::maxwell_boltzmann);
}

TEST(Generator, MoveAssignReplacesActiveLeaf) {
    Generator target(make_uniform());
    Generator source(make_maxwell_boltzmann());

    target = std::move(source);

    EXPECT_EQ(target.type, GeneratorType::maxwell_boltzmann);

    FluidVelocityState velocities(4);
    FluidSpeciesState  species(4);
    EXPECT_EQ(target.generate(&velocities, &species, 0, 4), 4);
    EXPECT_EQ(species.data()[0], std::size_t { 4 });
}

TEST(Generator, GenerateWithZeroCountIsNoOp) {
    const Generator    generator(make_uniform());
    FluidVelocityState velocities(4);
    FluidSpeciesState  species(4);

    EXPECT_EQ(generator.generate(&velocities, &species, 0, 0), 0);
}
