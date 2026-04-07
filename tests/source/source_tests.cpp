#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <atlas/source/source.h>
#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <vector>

using namespace atlas;

namespace {

template <typename T>
atlas::FluidHostPtr<T>
make_test_fluid() {
    const auto species_a = atlas::system::FluidicParticle<T>::builder()
                               .with_molecular_mass(T(1))
                               .make_host_shared();
    const auto species_b = atlas::system::FluidicParticle<T>::builder()
                               .with_molecular_mass(T(2))
                               .make_host_shared();

    return atlas::system::Fluid<T>::builder()
        .add_species(species_a, T(0.5))
        .add_species(species_b, T(0.5))
        .make_host_shared();
}

template <typename T>
atlas::Unit<T>
make_box_unit(const atlas::Vector3<T>& translation = atlas::Vector3<T>(T(0), T(0), T(0))) {
    const auto geometry = atlas::geometry::Box<T>::builder()
                              .with_lower_corner(atlas::Vector3<T>(T(-1), T(-1), T(-1)))
                              .with_upper_corner(atlas::Vector3<T>(T(1), T(1), T(1)))
                              .make_host_shared();

    const auto sync = atlas::system::Sync<T>::builder()
                          .with_rigid_pose(translation, atlas::Quaternion<T>())
                          .make_host_shared();

    return atlas::system::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .build();
}

}

TEST(Source, BuilderBuildStoresConfiguredValues) {
    const auto fluid = make_test_fluid<double>();
    const auto unit  = make_box_unit<double>(Vector3<double>(3.0, 4.0, 5.0));
    const auto generator = atlas::UniformGenerator<double>::builder()
                               .with_min_value(-1.0)
                               .with_max_value(2.0)
                               .with_seed(17u)
                               .make_host_shared();

    const auto source = atlas::Source<double>::builder()
                            .with_unit(unit)
                            .with_fluid(fluid)
                            .with_spawn_type(atlas::system::SpawnType::Volume)
                            .with_tolerance(0.25)
                            .with_spacing(0.5)
                            .with_temperature(425.0)
                            .with_generator(generator)
                            .build();

    EXPECT_EQ(source.fluid(), fluid);
    EXPECT_EQ(source.spawn_type(), atlas::system::SpawnType::Volume);
    EXPECT_TRUE(test::near(source.tolerance(), 0.25, 1e-12));
    EXPECT_TRUE(test::near(source.spacing(), 0.5, 1e-12));
    EXPECT_TRUE(test::near(source.temperature(), 425.0, 1e-12));
    ASSERT_TRUE(source.generator());
    EXPECT_EQ(source.generator()->type(), atlas::GenerateType::uniform);
    EXPECT_TRUE(test::near(source.generator()->param0(), -1.0, 1e-12));
    EXPECT_TRUE(test::near(source.generator()->param1(), 2.0, 1e-12));
    EXPECT_TRUE(test::vec_near(
        source.unit().sync_operator().translation,
        Vector3<double>(3.0, 4.0, 5.0),
        1e-12));
}

TEST(Source, BuilderRejectsMissingRequiredInputsAndInvalidSpacing) {
    const auto fluid = make_test_fluid<double>();
    const auto unit  = make_box_unit<double>();

    EXPECT_THROW(
        atlas::Source<double>::builder()
            .with_fluid(fluid)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::Source<double>::builder()
            .with_unit(unit)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::Source<double>::builder()
            .with_unit(unit)
            .with_fluid(fluid)
            .with_spacing(0.0)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::Source<double>::builder()
            .with_unit(unit)
            .with_fluid(fluid)
            .with_temperature(-1.0)
            .build(),
        std::runtime_error);
}

TEST(Source, EmitCachesSpawnableLocalPositionsAndWritesWorldParticles) {
    constexpr double eps = 1e-12;
    constexpr double source_temperature = 350.0;

    auto system          = atlas::system::System<double>(64);
    auto& probe          = system.particle_probe();
    probe.particle_count = 0;

    auto source = atlas::Source<double>::builder()
                      .with_unit(make_box_unit<double>(Vector3<double>(10.0, 0.0, -2.0)))
                      .with_fluid(make_test_fluid<double>())
                      .with_spawn_type(atlas::system::SpawnType::Volume)
                      .with_tolerance(0.0)
                      .with_spacing(1.0)
                      .with_temperature(source_temperature)
                      .build();

    source.emit(probe);

    ASSERT_EQ(source.local_positions().size(), 27u);
    EXPECT_EQ(probe.particle_count, 27);

    const auto local_positions = test::copy_device_buffer(source.local_positions());
    const auto world_positions = test::copy_device_range(probe.pos, static_cast<std::size_t>(probe.particle_count));
    const auto velocities      = test::copy_device_range(probe.vel, static_cast<std::size_t>(probe.particle_count));
    const auto temperatures    = test::copy_device_range(probe.temperature, static_cast<std::size_t>(probe.particle_count));
    const auto species         = test::copy_device_range(probe.species, static_cast<std::size_t>(probe.particle_count));

    ASSERT_EQ(local_positions.size(), world_positions.size());
    ASSERT_EQ(world_positions.size(), velocities.size());
    ASSERT_EQ(velocities.size(), temperatures.size());
    ASSERT_EQ(velocities.size(), species.size());

    const auto query = source.unit().geometry_operator();
    for (std::size_t i = 0; i < local_positions.size(); ++i) {
        EXPECT_TRUE(query.is_inside(local_positions[i], 0.0));
        EXPECT_TRUE(test::vec_near(
            world_positions[i],
            local_positions[i] + Vector3<double>(10.0, 0.0, -2.0),
            eps));
        EXPECT_TRUE(test::is_finite_vec(velocities[i]));
        EXPECT_TRUE(test::points_in_range(std::array<Vector3<double>, 1> { velocities[i] }, 0.0, 1.0));
        EXPECT_TRUE(test::near(temperatures[i], source_temperature, eps));
        EXPECT_LT(species[i], std::size_t(2));
    }

    const auto species_zero_count = static_cast<int>(std::count(species.begin(), species.end(), std::size_t(0)));
    const auto species_one_count  = static_cast<int>(std::count(species.begin(), species.end(), std::size_t(1)));
    EXPECT_EQ(species_zero_count, 14);
    EXPECT_EQ(species_one_count, 13);
}

TEST(Source, EmitAppendsUsingCachedPositionsAndPreservesSpeciesTotalsPerEmission) {
    auto system          = atlas::system::System<double>(64);
    auto& probe          = system.particle_probe();
    probe.particle_count = 0;

    auto source = atlas::Source<double>::builder()
                      .with_unit(make_box_unit<double>())
                      .with_fluid(make_test_fluid<double>())
                      .with_spawn_type(atlas::system::SpawnType::Volume)
                      .with_spacing(1.0)
                      .build();

    source.emit(probe);
    ASSERT_EQ(probe.particle_count, 27);

    const auto first_species = test::copy_device_range(probe.species, 27);
    source.emit(probe);

    ASSERT_EQ(probe.particle_count, 54);
    const auto all_species = test::copy_device_range(probe.species, 54);
    const std::vector<std::size_t> second_species(all_species.begin() + 27, all_species.end());

    EXPECT_EQ(std::count(first_species.begin(), first_species.end(), std::size_t(0)), 14);
    EXPECT_EQ(std::count(first_species.begin(), first_species.end(), std::size_t(1)), 13);
    EXPECT_EQ(std::count(second_species.begin(), second_species.end(), std::size_t(0)), 14);
    EXPECT_EQ(std::count(second_species.begin(), second_species.end(), std::size_t(1)), 13);
    EXPECT_EQ(source.local_positions().size(), 27u);
}

TEST(Source, ChangingSpawnTypeInvalidatesCacheAndRebuildsLocalPositions) {
    auto system          = atlas::system::System<double>(64);
    auto& probe          = system.particle_probe();
    probe.particle_count = 0;

    auto source = atlas::Source<double>::builder()
                      .with_unit(make_box_unit<double>())
                      .with_fluid(make_test_fluid<double>())
                      .with_spawn_type(atlas::system::SpawnType::Volume)
                      .with_spacing(1.0)
                      .build();

    source.emit(probe);
    ASSERT_EQ(source.local_positions().size(), 27u);

    probe.particle_count = 0;
    source.set_spawn_type(atlas::system::SpawnType::Surface);
    source.emit(probe);

    EXPECT_EQ(source.local_positions().size(), 26u);
    EXPECT_EQ(probe.particle_count, 26);

    const auto local_positions = test::copy_device_buffer(source.local_positions());
    const auto query           = source.unit().geometry_operator();
    for (const auto& local_position : local_positions) {
        EXPECT_TRUE(query.is_on_surface(local_position, 0.0));
    }
}

TEST(Source, FlipInvertsSpawnClassificationWhenRebuildingCache) {
    auto system          = atlas::system::System<double>(64);
    auto& probe          = system.particle_probe();
    probe.particle_count = 0;

    auto source = atlas::Source<double>::builder()
                      .with_unit(make_box_unit<double>())
                      .with_fluid(make_test_fluid<double>())
                      .with_spawn_type(atlas::system::SpawnType::Volume)
                      .with_flip(true)
                      .with_spacing(1.0)
                      .build();

    source.emit(probe);

    EXPECT_TRUE(source.flip());
    EXPECT_EQ(source.local_positions().size(), 0u);
    EXPECT_EQ(probe.particle_count, 0);

    const auto local_positions = test::copy_device_buffer(source.local_positions());
    const auto query           = source.unit().geometry_operator();
    for (const auto& local_position : local_positions) {
        EXPECT_FALSE(query.is_inside(local_position, 0.0));
    }
}
