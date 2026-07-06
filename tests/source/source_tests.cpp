#include <atlas/source/source.h>

#include <atlas/fluid/fluid.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/box.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace {

atlas::FluidHostPtr
make_fluid(const std::size_t buffer_size = 8) {
    atlas::HostBuffer<atlas::MaterialProperties> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr> generators(1);

    properties[0] = atlas::MaterialProperties::builder()
                        .with_type(atlas::MaterialType::molecule)
                        .with_mass(4.651734e-26f)
                        .with_molecular_mass(4.651734e-26f)
                        .with_species_id(0)
                        .with_reference_diameter(4.17e-10f)
                        .build();

    generators[0] = atlas::MaxwellBoltzmannGenerator::builder()
                        .with_temperature(300.0f)
                        .with_molecular_mass(4.651734e-26f)
                        .with_bulk_velocity(atlas::Vector3(0.0f, 0.0f, 0.0f))
                        .with_seed(7u)
                        .make_host_shared();

    return atlas::Fluid::builder()
        .with_buffer_size(buffer_size)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

atlas::Unit
make_unit() {
    const auto geometry = atlas::Box::builder()
                              .with_lower_corner(atlas::Vector3(-1.0f, -1.0f, -1.0f))
                              .with_upper_corner(atlas::Vector3(1.0f, 1.0f, 1.0f))
                              .make_host_shared();

    const auto sync = atlas::Sync::builder()
                          .make_host_shared();

    return atlas::Unit::builder()
        .with_geometry(atlas::Geometry(*geometry))
        .with_sync(sync)
        .build();
}

atlas::Spawn
make_spawn_operator() {
    return atlas::Spawn(atlas::SpawnType::surface);
}

atlas::UniverseHostPtr
make_universe(const atlas::HostBuffer<atlas::Unit>& source_units) {
    return atlas::Universe::builder()
        .with_lower_corner(atlas::Vector3(-10.0f, -10.0f, -10.0f))
        .with_upper_corner(atlas::Vector3(10.0f, 10.0f, 10.0f))
        .with_cell_size(1.0f)
        .with_source_units(source_units)
        .make_host_shared();
}

atlas::HostBuffer<atlas::Unit>
make_units(const std::size_t count = 1) {
    atlas::HostBuffer<atlas::Unit> units(count);
    for (std::size_t i = 0; i < count; ++i) {
        units[i] = make_unit();
    }
    return units;
}

atlas::HostBuffer<atlas::SpawnType>
make_spawn_types(const std::size_t count, const atlas::SpawnType type) {
    return atlas::HostBuffer<atlas::SpawnType>(count, type);
}

atlas::HostBuffer<atlas::Spawn>
make_spawn_operators(const std::size_t count) {
    return atlas::HostBuffer<atlas::Spawn>(count, make_spawn_operator());
}

std::size_t
box_axis_sample_count() {
    constexpr float lower   = -1.0f;
    constexpr float upper   = 1.0f;
    constexpr float spacing = 1.0f;
    return static_cast<std::size_t>(std::floor((upper - lower) / spacing)) + 1u;
}

}

TEST(Source, BuilderConstructsUsableSource) {
    const auto fluid = make_fluid();

    auto source = atlas::Source::builder()
                      .with_universe(make_universe(make_units()))
                      .with_fluid(fluid)
                      .with_spawn_types(make_spawn_types(1, atlas::SpawnType::surface))
                      .with_spawn_operator(make_spawn_operator())
                      .with_spacing(0.5f)
                      .with_tolerance(0.1f)
                      .with_temperature(300.0f)
                      .with_flip(true)
                      .build();

    EXPECT_NO_THROW(source.update(0.1f));
    EXPECT_NO_THROW(source.rebuild_cache());
    EXPECT_NO_THROW(source.emit());
}

TEST(Source, BuilderRejectsMissingDependencies) {
    const auto fluid = make_fluid();

    EXPECT_THROW(static_cast<void>(atlas::Source::builder()
                                       .with_fluid(fluid)
                                       .with_spawn_types(make_spawn_types(1, atlas::SpawnType::surface))
                                       .with_spawn_operator(make_spawn_operator())
                                       .build()),
                 std::runtime_error);

    EXPECT_THROW(static_cast<void>(atlas::Source::builder()
                                       .with_universe(make_universe(make_units()))
                                       .with_spawn_types(make_spawn_types(1, atlas::SpawnType::surface))
                                       .with_spawn_operator(make_spawn_operator())
                                       .build()),
                 std::runtime_error);

    EXPECT_THROW(static_cast<void>(atlas::Source::builder()
                                       .with_universe(make_universe(atlas::HostBuffer<atlas::Unit> {}))
                                       .with_fluid(fluid)
                                       .with_spawn_types(make_spawn_types(1, atlas::SpawnType::surface))
                                       .with_spawn_operator(make_spawn_operator())
                                       .build()),
                 std::runtime_error);
}

TEST(Source, BuilderRejectsMismatchedSpawnConfigurationSizes) {
    const auto fluid = make_fluid();

    EXPECT_THROW(static_cast<void>(atlas::Source::builder()
                                       .with_universe(make_universe(make_units(2)))
                                       .with_fluid(fluid)
                                       .with_spawn_types(make_spawn_types(3, atlas::SpawnType::surface))
                                       .with_spawn_operator(make_spawn_operator())
                                       .build()),
                 std::runtime_error);

    EXPECT_THROW(static_cast<void>(atlas::Source::builder()
                                       .with_universe(make_universe(make_units(2)))
                                       .with_fluid(fluid)
                                       .with_spawn_types(make_spawn_types(1, atlas::SpawnType::surface))
                                       .with_spawn_operators(make_spawn_operators(3))
                                       .build()),
                 std::runtime_error);
}

TEST(Source, BuilderRejectsInvalidImmediateInputs) {
    EXPECT_THROW(atlas::Source::builder()
                     .with_spawn_types(atlas::HostBuffer<atlas::SpawnType> {}),
                 std::runtime_error);

    EXPECT_THROW(atlas::Source::builder()
                     .with_spawn_operators(atlas::HostBuffer<atlas::Spawn> {}),
                 std::runtime_error);
}

TEST(Source, BuilderRejectsInvalidNumericConfiguration) {
    const auto fluid = make_fluid();

    EXPECT_THROW(static_cast<void>(atlas::Source::builder()
                                       .with_universe(make_universe(make_units()))
                                       .with_fluid(fluid)
                                       .with_spawn_types(make_spawn_types(1, atlas::SpawnType::surface))
                                       .with_spawn_operator(make_spawn_operator())
                                       .with_spacing(0.0f)
                                       .build()),
                 std::runtime_error);

    EXPECT_THROW(static_cast<void>(atlas::Source::builder()
                                       .with_universe(make_universe(make_units()))
                                       .with_fluid(fluid)
                                       .with_spawn_types(make_spawn_types(1, atlas::SpawnType::surface))
                                       .with_spawn_operator(make_spawn_operator())
                                       .with_tolerance(std::numeric_limits<float>::infinity())
                                       .build()),
                 std::runtime_error);
}

TEST(Source, MakeHostSharedBuildsSource) {
    const auto fluid = make_fluid();

    const auto source = atlas::Source::builder()
                            .with_universe(make_universe(make_units()))
                            .with_fluid(fluid)
                            .with_spawn_types(make_spawn_types(1, atlas::SpawnType::surface))
                            .with_spawn_operator(make_spawn_operator())
                            .make_host_shared();

    ASSERT_NE(source, nullptr);
    EXPECT_NO_THROW(source->update(0.1f));
}

TEST(Source, UpdateIgnoresNonPositiveDt) {
    const auto fluid = make_fluid();

    auto source = atlas::Source::builder()
                      .with_universe(make_universe(make_units()))
                      .with_fluid(fluid)
                      .with_spawn_types(make_spawn_types(1, atlas::SpawnType::surface))
                      .with_spawn_operator(make_spawn_operator())
                      .build();

    EXPECT_NO_THROW(source.update(0.0f));
}

TEST(Source, EmitIncreasesParticleCount) {
    const auto fluid = make_fluid();

    auto source = atlas::Source::builder()
                      .with_universe(make_universe(make_units()))
                      .with_fluid(fluid)
                      .with_spawn_types(make_spawn_types(1, atlas::SpawnType::volume))
                      .with_spawn_operator(atlas::Spawn(atlas::SpawnType::volume))
                      .with_spacing(1.0f)
                      .with_temperature(300.0f)
                      .build();

    source.emit();

    EXPECT_GT(fluid->particle_count(), 0u);
    EXPECT_LE(fluid->particle_count(), fluid->buffer_size());
}

TEST(Source, EmitMatchesExpectedVolumeParticleCountForBoxSpacing) {
    constexpr float spacing                    = 1.0f;
    const std::size_t samples_per_axis         = box_axis_sample_count();
    const std::size_t expected_particle_count  = samples_per_axis * samples_per_axis * samples_per_axis;
    const auto fluid                           = make_fluid(expected_particle_count);

    auto source = atlas::Source::builder()
                      .with_universe(make_universe(make_units()))
                      .with_fluid(fluid)
                      .with_spawn_types(make_spawn_types(1, atlas::SpawnType::volume))
                      .with_spawn_operator(atlas::Spawn(atlas::SpawnType::volume))
                      .with_spacing(spacing)
                      .with_temperature(300.0f)
                      .build();

    source.emit();

    EXPECT_EQ(expected_particle_count, 27u);
    EXPECT_EQ(fluid->particle_count(), expected_particle_count);
}

TEST(Source, EmitMatchesExpectedSurfaceParticleCountForBoxSpacing) {
    constexpr float spacing                    = 1.0f;
    const std::size_t samples_per_axis         = box_axis_sample_count();
    const std::size_t volume_sample_count      = samples_per_axis * samples_per_axis * samples_per_axis;
    const std::size_t interior_sample_count    = (samples_per_axis - 2u) * (samples_per_axis - 2u) * (samples_per_axis - 2u);
    const std::size_t expected_particle_count  = volume_sample_count - interior_sample_count;
    const auto fluid                           = make_fluid(expected_particle_count);

    auto source = atlas::Source::builder()
                      .with_universe(make_universe(make_units()))
                      .with_fluid(fluid)
                      .with_spawn_types(make_spawn_types(1, atlas::SpawnType::surface))
                      .with_spawn_operator(atlas::Spawn(atlas::SpawnType::surface))
                      .with_spacing(spacing)
                      .with_tolerance(0.0f)
                      .with_temperature(300.0f)
                      .build();

    source.emit();

    EXPECT_EQ(expected_particle_count, 26u);
    EXPECT_EQ(fluid->particle_count(), expected_particle_count);
}
