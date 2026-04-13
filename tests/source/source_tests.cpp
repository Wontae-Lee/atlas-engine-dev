#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <atlas/source/source.h>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

using namespace atlas;

TEST(Source, BuilderBuildStoresConfiguredValues) {
    const auto fluid = test::make_test_fluid<double>();
    const auto unit  = test::make_box_unit<double>(Vector3<double>(3.0, 4.0, 5.0));

    const auto source = atlas::Source<double>::builder()
                            .with_units({ unit })
                            .with_fluid(fluid)
                            .with_spawn_types({ atlas::system::SpawnType::Volume })
                            .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
                            .with_tolerance(0.25)
                            .with_spacing(0.5)
                            .with_temperature(425.0)
                            .build();

    EXPECT_EQ(source.fluid(), fluid);
    ASSERT_EQ(source.units().size(), 1u);
    ASSERT_EQ(source.spawn_types().size(), 1u);
    ASSERT_EQ(source.spawn_operators().size(), 1u);
    EXPECT_EQ(source.spawn_types()[0], atlas::system::SpawnType::Volume);
    EXPECT_EQ(source.spawn_operators()[0].type, atlas::system::SpawnType::Volume);
    EXPECT_TRUE(test::near(source.tolerance(), 0.25, eps));
    EXPECT_TRUE(test::near(source.spacing(), 0.5, eps));
    EXPECT_TRUE(test::near(source.temperature(), 425.0, eps));
    EXPECT_TRUE(test::vec_near(
        source.units()[0].sync_operator().translation,
        Vector3<double>(3.0, 4.0, 5.0),
        1e-12));
}

TEST(Source, BuilderRejectsMissingRequiredInputsAndInvalidSpacing) {
    const auto fluid = test::make_test_fluid<double>();
    const auto unit  = test::make_box_unit<double>();

    EXPECT_THROW(
        atlas::Source<double>::builder()
            .with_fluid(fluid)
            .with_spawn_types({ atlas::system::SpawnType::Volume })
            .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::Source<double>::builder()
            .with_units({ unit })
            .with_spawn_types({ atlas::system::SpawnType::Volume })
            .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::Source<double>::builder()
            .with_units({ unit })
            .with_fluid(fluid)
            .with_spawn_types({ atlas::system::SpawnType::Volume })
            .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
            .with_spacing(0.0)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::Source<double>::builder()
            .with_units({ unit })
            .with_fluid(fluid)
            .with_spawn_types({ atlas::system::SpawnType::Volume })
            .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
            .with_temperature(std::numeric_limits<double>::quiet_NaN())
            .build(),
        std::runtime_error);
}

TEST(Source, EmitCachesSpawnableLocalPositionsAndWritesWorldParticles) {
    constexpr double source_temperature = 350.0;

    auto system          = atlas::system::System<double>(test::make_buffered_fluid<double>(64));
    auto& probe          = system.particle_probe();
    probe.particle_count = 0;

    auto source = atlas::Source<double>::builder()
                      .with_units({ test::make_box_unit<double>(Vector3<double>(10.0, 0.0, -2.0)) })
                      .with_fluid(test::make_test_fluid<double>())
                      .with_spawn_types({ atlas::system::SpawnType::Volume })
                      .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
                      .with_tolerance(0.0)
                      .with_spacing(1.0)
                      .with_temperature(source_temperature)
                      .build();

    source.emit(probe);

    ASSERT_EQ(source.local_positions().size(), 1u);

    std::vector<Vector3<double>> local_positions;
    for (const auto& positions : source.local_positions()) {
        const auto copied = test::copy_device_buffer(positions);
        local_positions.insert(local_positions.end(), copied.begin(), copied.end());
    }

    EXPECT_EQ(local_positions.size(), 27u);
    EXPECT_EQ(probe.particle_count, 27);

    const auto world_positions = test::copy_device_range(probe.pos, static_cast<std::size_t>(probe.particle_count));
    const auto velocities      = test::copy_device_range(probe.vel, static_cast<std::size_t>(probe.particle_count));
    const auto temperatures    = test::copy_device_range(probe.temperature, static_cast<std::size_t>(probe.particle_count));
    const auto species         = test::copy_device_range(probe.species, static_cast<std::size_t>(probe.particle_count));

    ASSERT_EQ(local_positions.size(), world_positions.size());
    ASSERT_EQ(world_positions.size(), velocities.size());
    ASSERT_EQ(velocities.size(), temperatures.size());
    ASSERT_EQ(velocities.size(), species.size());

    const auto query = source.units()[0].geometry_operator();
    for (std::size_t i = 0; i < local_positions.size(); ++i) {
        EXPECT_TRUE(query.is_inside(local_positions[i], 0.0) || query.is_on_surface(local_positions[i], 0.0));
        EXPECT_TRUE(test::vec_near(
            world_positions[i],
            local_positions[i] + Vector3<double>(10.0, 0.0, -2.0),
            eps));
        EXPECT_TRUE(test::is_finite_vec(velocities[i]));
        EXPECT_GT(velocities[i].length(), 0.0);
        EXPECT_TRUE(test::near(temperatures[i], source_temperature, eps));
        EXPECT_LT(species[i], static_cast<std::size_t>(2));
    }

    const auto species_zero_count = static_cast<int>(std::count(species.begin(), species.end(), static_cast<std::size_t>(0)));
    const auto species_one_count  = static_cast<int>(std::count(species.begin(), species.end(), static_cast<std::size_t>(1)));
    EXPECT_EQ(species_zero_count, 14);
    EXPECT_EQ(species_one_count, 13);
}

TEST(Source, EmitAppendsUsingCachedPositionsAndPreservesSpeciesTotalsPerEmission) {
    auto system          = atlas::system::System<double>(test::make_buffered_fluid<double>(64));
    auto& probe          = system.particle_probe();
    probe.particle_count = 0;

    auto source = atlas::Source<double>::builder()
                      .with_units({ test::make_box_unit<double>() })
                      .with_fluid(test::make_test_fluid<double>())
                      .with_spawn_types({ atlas::system::SpawnType::Volume })
                      .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
                      .with_spacing(1.0)
                      .build();

    source.emit(probe);
    ASSERT_EQ(probe.particle_count, 27);

    const auto first_species = test::copy_device_range(probe.species, 27);

    source.emit(probe);

    ASSERT_EQ(probe.particle_count, 54);

    const auto all_species = test::copy_device_range(probe.species, 54);
    const std::vector<std::size_t> second_species(all_species.begin() + 27, all_species.end());

    EXPECT_EQ(std::count(first_species.begin(), first_species.end(), static_cast<std::size_t>(0)), 14);
    EXPECT_EQ(std::count(first_species.begin(), first_species.end(), static_cast<std::size_t>(1)), 13);
    EXPECT_EQ(std::count(second_species.begin(), second_species.end(), static_cast<std::size_t>(0)), 14);
    EXPECT_EQ(std::count(second_species.begin(), second_species.end(), static_cast<std::size_t>(1)), 13);

    EXPECT_EQ(source.local_positions().size(), 1u);
    EXPECT_EQ(source.local_positions()[0].size(), 27u);
}

TEST(Source, SurfaceSpawnBuildCreatesSurfaceOnlyLocalPositions) {
    auto system          = atlas::system::System<double>(test::make_buffered_fluid<double>(64));
    auto& probe          = system.particle_probe();
    probe.particle_count = 0;

    auto volume_source = atlas::Source<double>::builder()
                             .with_units({ test::make_box_unit<double>() })
                             .with_fluid(test::make_test_fluid<double>())
                             .with_spawn_types({ atlas::system::SpawnType::Volume })
                             .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
                             .with_spacing(1.0)
                             .build();

    volume_source.emit(probe);

    ASSERT_EQ(volume_source.local_positions().size(), 1u);

    std::vector<Vector3<double>> volume_local_positions;
    for (const auto& positions : volume_source.local_positions()) {
        const auto copied = test::copy_device_buffer(positions);
        volume_local_positions.insert(
            volume_local_positions.end(),
            copied.begin(),
            copied.end());
    }

    ASSERT_EQ(volume_local_positions.size(), 27u);

    probe.particle_count = 0;

    auto surface_source = atlas::Source<double>::builder()
                              .with_units({ test::make_box_unit<double>() })
                              .with_fluid(test::make_test_fluid<double>())
                              .with_spawn_types({ atlas::system::SpawnType::Surface })
                              .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Surface))
                              .with_spacing(1.0)
                              .build();

    surface_source.emit(probe);

    ASSERT_EQ(surface_source.local_positions().size(), 1u);
    EXPECT_EQ(probe.particle_count, 26);

    std::vector<Vector3<double>> local_positions;
    for (const auto& positions : surface_source.local_positions()) {
        const auto copied = test::copy_device_buffer(positions);
        local_positions.insert(local_positions.end(), copied.begin(), copied.end());
    }

    EXPECT_EQ(local_positions.size(), 26u);

    const auto query = surface_source.units()[0].geometry_operator();
    for (const auto& local_position : local_positions) {
        EXPECT_TRUE(query.is_on_surface(local_position, 0.0));
    }
}

TEST(Source, FlipInvertsSpawnClassificationWhenRebuildingCache) {
    auto system          = atlas::system::System<double>(test::make_buffered_fluid<double>(64));
    auto& probe          = system.particle_probe();
    probe.particle_count = 0;

    auto source = atlas::Source<double>::builder()
                      .with_units({ test::make_box_unit<double>() })
                      .with_fluid(test::make_test_fluid<double>())
                      .with_spawn_types({ atlas::system::SpawnType::Volume })
                      .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
                      .with_flip(true)
                      .with_spacing(1.0)
                      .build();

    source.emit(probe);

    EXPECT_TRUE(source.flip());
    ASSERT_EQ(source.local_positions().size(), 1u);
    EXPECT_EQ(probe.particle_count, 0);

    std::vector<Vector3<double>> local_positions;
    for (const auto& positions : source.local_positions()) {
        const auto copied = test::copy_device_buffer(positions);
        local_positions.insert(local_positions.end(), copied.begin(), copied.end());
    }

    EXPECT_TRUE(local_positions.empty());

    const auto query = source.units()[0].geometry_operator();
    for (const auto& local_position : local_positions) {
        EXPECT_FALSE(query.is_inside(local_position, 0.0));
    }
}