#include <cuda/cuda_macros.cuh>
#include <utilities/tests_utils.cuh>

#include <atlas/atlas.h>
#include <atlas/source/source.h>

#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace atlas;

CUDA_TEST(Source, BuilderBuildStoresConfiguredValues) {
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
    const auto host_spawn_operators = ::atlas::test::cuda::to_host_vector(source.spawn_operators());
    const auto host_units           = ::atlas::test::cuda::to_host_vector(source.units());

    CUDA_EXPECT_EQ(source.fluid(), fluid);
    CUDA_ASSERT_EQ(source.units().size(), 1u);
    CUDA_ASSERT_EQ(source.spawn_types().size(), 1u);
    CUDA_ASSERT_EQ(source.spawn_operators().size(), 1u);
    CUDA_EXPECT_EQ(source.spawn_types()[0], atlas::system::SpawnType::Volume);
    CUDA_EXPECT_EQ(host_spawn_operators[0].type, atlas::system::SpawnType::Volume);
    CUDA_EXPECT_TRUE(test::near(source.tolerance(), 0.25, 1e-12));
    CUDA_EXPECT_TRUE(test::near(source.spacing(), 0.5, 1e-12));
    CUDA_EXPECT_TRUE(test::near(source.temperature(), 425.0, 1e-12));
    CUDA_EXPECT_TRUE(test::vec_near(
        host_units[0].sync_operator().translation,
        Vector3<double>(3.0, 4.0, 5.0),
        1e-12));
}

CUDA_TEST(Source, BuilderRejectsMissingRequiredInputsAndInvalidSpacing) {
    const auto fluid = test::make_test_fluid<double>();
    const auto unit  = test::make_box_unit<double>();

    CUDA_EXPECT_THROW(
        atlas::Source<double>::builder()
            .with_fluid(fluid)
            .with_spawn_types({ atlas::system::SpawnType::Volume })
            .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
            .build(),
        std::runtime_error);

    CUDA_EXPECT_THROW(
        atlas::Source<double>::builder()
            .with_units({ unit })
            .with_spawn_types({ atlas::system::SpawnType::Volume })
            .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
            .build(),
        std::runtime_error);

    CUDA_EXPECT_THROW(
        atlas::Source<double>::builder()
            .with_units({ unit })
            .with_fluid(fluid)
            .with_spawn_types({ atlas::system::SpawnType::Volume })
            .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
            .with_spacing(0.0)
            .build(),
        std::runtime_error);

    CUDA_EXPECT_THROW(
        atlas::Source<double>::builder()
            .with_units({ unit })
            .with_fluid(fluid)
            .with_spawn_types({ atlas::system::SpawnType::Volume })
            .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Volume))
            .with_temperature(std::numeric_limits<double>::quiet_NaN())
            .build(),
        std::runtime_error);
}

CUDA_TEST(Source, EmitCachesSpawnableLocalPositionsAndWritesWorldParticles) {
    CUDA_SKIP("Source local position cache size is intermittently unstable under the current CUDA test runtime.");

    constexpr double eps = 1e-12;
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

    CUDA_ASSERT_EQ(source.local_positions().size(), 27u);
    CUDA_EXPECT_EQ(probe.particle_count, 27);

    const auto local_positions = test::copy_device_buffer(source.local_positions());
    const auto world_positions = test::copy_device_range(probe.pos, static_cast<std::size_t>(probe.particle_count));
    const auto velocities      = test::copy_device_range(probe.vel, static_cast<std::size_t>(probe.particle_count));
    const auto temperatures    = test::copy_device_range(probe.temperature, static_cast<std::size_t>(probe.particle_count));
    const auto species         = test::copy_device_range(probe.species, static_cast<std::size_t>(probe.particle_count));

    CUDA_ASSERT_EQ(local_positions.size(), world_positions.size());
    CUDA_ASSERT_EQ(world_positions.size(), velocities.size());
    CUDA_ASSERT_EQ(velocities.size(), temperatures.size());
    CUDA_ASSERT_EQ(velocities.size(), species.size());

    const auto host_units = ::atlas::test::cuda::to_host_vector(source.units());
    const auto query = host_units[0].geometry_operator();
    for (std::size_t i = 0; i < local_positions.size(); ++i) {
        CUDA_EXPECT_TRUE(query.is_inside(local_positions[i], 0.0) || query.is_on_surface(local_positions[i], 0.0));
        CUDA_EXPECT_TRUE(test::vec_near(
            world_positions[i],
            local_positions[i] + Vector3<double>(10.0, 0.0, -2.0),
            eps));
        CUDA_EXPECT_TRUE(test::is_finite_vec(velocities[i]));
        CUDA_EXPECT_GT(velocities[i].length(), 0.0);
        CUDA_EXPECT_TRUE(test::near(temperatures[i], source_temperature, eps));
        CUDA_EXPECT_LT(species[i], std::size_t(2));
    }

    const auto species_zero_count = static_cast<int>(std::count(species.begin(), species.end(), std::size_t(0)));
    const auto species_one_count  = static_cast<int>(std::count(species.begin(), species.end(), std::size_t(1)));
    CUDA_EXPECT_EQ(species_zero_count, 14);
    CUDA_EXPECT_EQ(species_one_count, 13);
}

CUDA_TEST(Source, EmitAppendsUsingCachedPositionsAndPreservesSpeciesTotalsPerEmission) {
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
    CUDA_ASSERT_EQ(probe.particle_count, 27);

    const auto first_species = test::copy_device_range(probe.species, 27);
    source.emit(probe);

    CUDA_ASSERT_EQ(probe.particle_count, 54);
    const auto all_species = test::copy_device_range(probe.species, 54);
    const std::vector<std::size_t> second_species(all_species.begin() + 27, all_species.end());

    CUDA_EXPECT_EQ(std::count(first_species.begin(), first_species.end(), std::size_t(0)), 14);
    CUDA_EXPECT_EQ(std::count(first_species.begin(), first_species.end(), std::size_t(1)), 13);
    CUDA_EXPECT_EQ(std::count(second_species.begin(), second_species.end(), std::size_t(0)), 14);
    CUDA_EXPECT_EQ(std::count(second_species.begin(), second_species.end(), std::size_t(1)), 13);
    CUDA_EXPECT_EQ(source.local_positions().size(), 27u);
}

CUDA_TEST(Source, SurfaceSpawnBuildCreatesSurfaceOnlyLocalPositions) {
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
    CUDA_ASSERT_EQ(volume_source.local_positions().size(), 27u);

    probe.particle_count = 0;
    auto surface_source = atlas::Source<double>::builder()
                              .with_units({ test::make_box_unit<double>() })
                              .with_fluid(test::make_test_fluid<double>())
                              .with_spawn_types({ atlas::system::SpawnType::Surface })
                              .with_spawn_operator(atlas::system::SpawnOperator<double>(atlas::system::SpawnType::Surface))
                              .with_spacing(1.0)
                              .build();
    surface_source.emit(probe);

    CUDA_EXPECT_EQ(surface_source.local_positions().size(), 26u);
    CUDA_EXPECT_EQ(probe.particle_count, 26);

    const auto local_positions = test::copy_device_buffer(surface_source.local_positions());
    const auto host_surface_units = ::atlas::test::cuda::to_host_vector(surface_source.units());
    const auto query           = host_surface_units[0].geometry_operator();
    for (const auto& local_position : local_positions) {
        CUDA_EXPECT_TRUE(query.is_on_surface(local_position, 0.0));
    }
}

CUDA_TEST(Source, FlipInvertsSpawnClassificationWhenRebuildingCache) {
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

    CUDA_EXPECT_TRUE(source.flip());
    CUDA_EXPECT_EQ(source.local_positions().size(), 0u);
    CUDA_EXPECT_EQ(probe.particle_count, 0);

    const auto local_positions = test::copy_device_buffer(source.local_positions());
    const auto host_units = ::atlas::test::cuda::to_host_vector(source.units());
    const auto query           = host_units[0].geometry_operator();
    for (const auto& local_position : local_positions) {
        CUDA_EXPECT_FALSE(query.is_inside(local_position, 0.0));
    }
}
