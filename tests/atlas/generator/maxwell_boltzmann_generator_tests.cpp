#include <atlas/generator/maxwell_boltzmann_generator.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

using atlas::Float3;
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

TEST(MaxwellBoltzmannGenerator, BuilderNormalizesUnnormalizedRatios) {
    // Unnormalized weights {70, 30} must be rescaled to {0.7, 0.3} at build() so
    // both species are actually drawn. Without normalization sample_weighted_index
    // returns index 0 for every draw (u in [0, 1) is always <= the cumulative 70),
    // so species 1 would never appear.
    const auto generator = MaxwellBoltzmannGenerator::builder()
                               .with_species_ratios({ 70.0f, 30.0f })
                               .with_species_numbers({ 0.0f, 1.0f })
                               .with_species_mass({ 2.0f, 3.0f })
                               .with_temperature(300.0f)
                               .with_seed(11u)
                               .build();

    const std::size_t  count = 8192;
    FluidVelocityState velocities(count);
    FluidSpeciesState  species(count);
    ASSERT_EQ(generator.generate(&velocities, &species, 0, count), static_cast<int>(count));

    std::size_t species0 = 0;
    std::size_t species1 = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t id = species.data()[i];
        if (id == 0) {
            ++species0;
        } else if (id == 1) {
            ++species1;
        }
    }

    // Both species appear (the pre-fix bug left species1 == 0), near the 70/30 split.
    EXPECT_GT(species0, std::size_t { 0 });
    EXPECT_GT(species1, std::size_t { 0 });
    const double fraction1 = static_cast<double>(species1) / static_cast<double>(count);
    EXPECT_GT(fraction1, 0.2);
    EXPECT_LT(fraction1, 0.4);
}

TEST(MaxwellBoltzmannGenerator, GenerateRejectsNullState) {
    const auto        generator = make_generator_with_direct_mass();
    FluidSpeciesState species(4);

    EXPECT_EQ(generator.generate(nullptr, &species, 0, 4), 0);
}

TEST(MaxwellBoltzmannGenerator, GenerateWithZeroCountIsNoOp) {
    const auto         generator = make_generator_with_direct_mass();
    FluidVelocityState velocities(4);
    FluidSpeciesState  species(4);

    EXPECT_EQ(generator.generate(&velocities, &species, 0, 0), 0);
}

TEST(MaxwellBoltzmannGenerator, SampledSpeedsMatchThermalSpeed) {
    // T = 300 K, m = 2, no bulk drift. The per-component sigma is
    // sqrt(k_B * T / m) and the mean speed of an isotropic Gaussian is
    // sigma * sqrt(8 / pi). Every speed must be finite and non-negative, and the
    // sample mean must land in a generous band around that thermal speed.
    const auto         generator = make_generator_with_direct_mass();
    const std::size_t  count     = 4096;
    FluidVelocityState velocities(count);
    FluidSpeciesState  species(count);

    ASSERT_EQ(generator.generate(&velocities, &species, 0, count), static_cast<int>(count));

    const double sigma        = std::sqrt(atlas::boltzmann_constant * 300.0 / 2.0);
    const double expected_mean = sigma * std::sqrt(8.0 / atlas::pi);

    double sum_speed = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const Float3 v     = velocities.data()[i];
        ASSERT_TRUE(atlas::isfinite(v));
        const double speed = std::sqrt(static_cast<double>(v.x) * v.x
                                       + static_cast<double>(v.y) * v.y
                                       + static_cast<double>(v.z) * v.z);
        EXPECT_GE(speed, 0.0);
        sum_speed += speed;
    }

    const double mean_speed = sum_speed / static_cast<double>(count);
    EXPECT_GT(mean_speed, 0.5 * expected_mean);
    EXPECT_LT(mean_speed, 1.5 * expected_mean);
}

TEST(MaxwellBoltzmannGenerator, ZeroTemperatureProducesBulkDriftOnly) {
    // A non-positive temperature yields a zero thermal sigma, so every particle is
    // assigned exactly the bulk drift and no random spread.
    const auto         generator = MaxwellBoltzmannGenerator::builder()
                               .with_species_ratios({ 1.0f })
                               .with_species_numbers({ 5.0f })
                               .with_species_mass({ 2.0f })
                               .with_temperature(0.0f)
                               .with_bulk_velocity(Float3(1.0f, 2.0f, 3.0f))
                               .with_seed(7u)
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

TEST(MaxwellBoltzmannGenerator, SetBulkVelocityUpdatesGetter) {
    auto generator = make_generator_with_direct_mass();

    generator.set_bulk_velocity(Float3(2.0f, -3.0f, 4.0f));

    const Float3 bulk = generator.bulk_velocity();
    EXPECT_NEAR(bulk.x, 2.0f, tol);
    EXPECT_NEAR(bulk.y, -3.0f, tol);
    EXPECT_NEAR(bulk.z, 4.0f, tol);
}
