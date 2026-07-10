#include <atlas/generator/maxwell_boltzmann_generator.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace {

using atlas::FluidSpeciesState;
using atlas::FluidVelocityState;
using atlas::Material;
using atlas::MaterialDictionary;
using atlas::MaxwellBoltzmannGenerator;
using atlas::Molecule;
using atlas::tol;

MaxwellBoltzmannGenerator
make_generator_with_direct_mass() {
    return MaxwellBoltzmannGenerator::builder()
        .with_species_ratios({ 1.0f })
        .with_species_numbers({ 5.0f })
        .with_species_mass({ 2.0f })
        .with_temperature(300.0f)
        .with_seed(7u)
        .build();
}

MaterialDictionary
make_dictionary(const float mass) {
    return MaterialDictionary::builder()
        .with_material(Material(Molecule(mass, 0.0f, 0.0f, 0.0f, 3.0e-10f, 273.0f, 0.5f, 1.0f)))
        .build();
}

}

TEST(MaxwellBoltzmannGenerator, BuilderRejectsEmptySpecies) {
    EXPECT_THROW(
        static_cast<void>(MaxwellBoltzmannGenerator::builder().with_species_mass({ 2.0f }).build()),
        std::runtime_error);
}

TEST(MaxwellBoltzmannGenerator, BuilderRejectsMissingMassSource) {
    EXPECT_THROW(
        static_cast<void>(MaxwellBoltzmannGenerator::builder().with_species_ratios({ 1.0f }).with_species_numbers({ 5.0f }).build()),
        std::runtime_error);
}

TEST(MaxwellBoltzmannGenerator, BuilderRejectsMassSizeMismatch) {
    EXPECT_THROW(
        static_cast<void>(MaxwellBoltzmannGenerator::builder()
                              .with_species_ratios({ 1.0f })
                              .with_species_numbers({ 5.0f })
                              .with_species_mass({ 2.0f, 3.0f })
                              .build()),
        std::runtime_error);
}

TEST(MaxwellBoltzmannGenerator, GenerateFillsStatesWithDirectMass) {
    const auto            generator = make_generator_with_direct_mass();
    const std::size_t     count     = 8;
    FluidVelocityState    velocities(count);
    FluidSpeciesState     species(count);

    const int filled = generator.generate(&velocities, &species, 0, count);

    EXPECT_EQ(filled, static_cast<int>(count));
    EXPECT_EQ(species.data()[0], std::size_t { 5 });
    EXPECT_TRUE(atlas::isfinite(velocities.data()[0]));
}

TEST(MaxwellBoltzmannGenerator, GenerateUsesDictionaryMass) {
    const auto generator = MaxwellBoltzmannGenerator::builder()
                               .with_species_ratios({ 1.0f })
                               .with_species_numbers({ 0.0f })
                               .with_material_dictionary(make_dictionary(2.0f))
                               .with_temperature(300.0f)
                               .with_seed(3u)
                               .build();

    FluidVelocityState    velocities(4);
    FluidSpeciesState     species(4);

    const int filled = generator.generate(&velocities, &species, 0, 4);

    EXPECT_EQ(filled, 4);
    EXPECT_EQ(species.data()[0], std::size_t { 0 });
    EXPECT_TRUE(atlas::isfinite(velocities.data()[0]));
}
