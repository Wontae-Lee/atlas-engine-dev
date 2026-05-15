#include "../utilities/test_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/geometry/box.h>
#include <atlas/source/source.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <testkit/testkit.h>

#include <cmath>
#include <limits>

namespace {

using atlas::Box;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::GeneratorHostPtr;
using atlas::HostBuffer;
using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::Unit;
using atlas::Vector3F;
using atlas::fluid::MaxwellBoltzmannGenerator;
using atlas::fluid::Source;
using atlas::fluid::SpawnOperator;
using atlas::fluid::SpawnType;
using atlas::physics::Sync;

FluidHostPtr<float>
make_fluid(const std::size_t buffer_size = 8) {
    HostBuffer<MaterialProperties<float>> properties(1);
    HostBuffer<GeneratorHostPtr<float>> generators(1);

    properties[0] = MaterialProperties<float>::builder()
                        .with_type(MaterialType::Molecule)
                        .with_mass(4.651734e-26f)
                        .with_molecular_mass(4.651734e-26f)
                        .with_species_id(0)
                        .with_reference_diameter(4.17e-10f)
                        .build();

    generators[0] = MaxwellBoltzmannGenerator<float>::builder()
                        .with_temperature(300.0f)
                        .with_molecular_mass(4.651734e-26f)
                        .with_bulk_velocity(Vector3F(0, 0, 0))
                        .with_seed(7u)
                        .make_host_shared();

    return Fluid<float>::builder()
        .with_buffer_size(buffer_size)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

Unit<float>
make_unit() {
    const auto geometry = Box<float>::builder()
                              .with_lower_corner(Vector3F(-1, -1, -1))
                              .with_upper_corner(Vector3F(1, 1, 1))
                              .make_host_shared();

    const auto sync = Sync<float>::builder()
                          .make_host_shared();

    return Unit<float>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .build();
}

SpawnOperator<float>
make_spawn_operator() {
    return SpawnOperator<float>(SpawnType::Surface);
}

std::size_t
box_axis_sample_count() {
    constexpr float lower = -1.0f;
    constexpr float upper = 1.0f;
    constexpr float spacing = 1.0f;
    return static_cast<std::size_t>(std::floor((upper - lower) / spacing)) + 1u;
}

} // namespace

TEST(Source, BuilderConstructsUsableSource) {
    // Arrange: create a fluid and fully configured source.
    const auto fluid = make_fluid();

    auto source = Source<float>::builder()
                      .with_units(HostBuffer<Unit<float>> { make_unit() })
                      .with_fluid(fluid)
                      .with_spawn_types(HostBuffer<SpawnType> { SpawnType::Surface })
                      .with_spawn_operator(make_spawn_operator())
                      .with_spacing(0.5f)
                      .with_tolerance(0.1f)
                      .with_temperature(300.0f)
                      .with_flip(true)
                      .build();

    // Assert: update, cache rebuild, and emission are callable on a valid source.
    EXPECT_NO_THROW(source.update(0.1f));
    EXPECT_NO_THROW(source.rebuild_cache());
    EXPECT_NO_THROW(source.emit());
}

TEST(Source, BuilderRejectsMissingDependencies) {
    // Arrange: create the shared fluid dependency.
    const auto fluid = make_fluid();

    // A source cannot be built without units.
    EXPECT_THROW(Source<float>::builder()
                     .with_fluid(fluid)
                     .with_spawn_types(HostBuffer<SpawnType> { SpawnType::Surface })
                     .with_spawn_operator(make_spawn_operator())
                     .build(),
                 std::runtime_error);

    // A source cannot be built without a fluid.
    EXPECT_THROW(Source<float>::builder()
                     .with_units(HostBuffer<Unit<float>> { make_unit() })
                     .with_spawn_types(HostBuffer<SpawnType> { SpawnType::Surface })
                     .with_spawn_operator(make_spawn_operator())
                     .build(),
                 std::runtime_error);
}

TEST(Source, BuilderRejectsMismatchedSpawnConfigurationSizes) {
    // Arrange: create the shared fluid dependency.
    const auto fluid = make_fluid();

    // Spawn type count must match unit count.
    EXPECT_THROW(Source<float>::builder()
                     .with_units(HostBuffer<Unit<float>> { make_unit(), make_unit() })
                     .with_fluid(fluid)
                     .with_spawn_types(HostBuffer<SpawnType> { SpawnType::Surface, SpawnType::Volume, SpawnType::Surface })
                     .with_spawn_operator(make_spawn_operator())
                     .build(),
                 std::runtime_error);

    // Spawn operator count must match unit count.
    EXPECT_THROW(Source<float>::builder()
                     .with_units(HostBuffer<Unit<float>> { make_unit(), make_unit() })
                     .with_fluid(fluid)
                     .with_spawn_types(HostBuffer<SpawnType> { SpawnType::Surface })
                     .with_spawn_operators(HostBuffer<SpawnOperator<float>> { make_spawn_operator(), make_spawn_operator(), make_spawn_operator() })
                     .build(),
                 std::runtime_error);
}

TEST(Source, BuilderRejectsInvalidImmediateInputs) {
    // Empty unit lists are rejected immediately.
    EXPECT_THROW(Source<float>::builder()
                     .with_units(HostBuffer<Unit<float>> {}),
                 std::runtime_error);

    // Empty spawn type lists are rejected immediately.
    EXPECT_THROW(Source<float>::builder()
                     .with_spawn_types(HostBuffer<SpawnType> {}),
                 std::runtime_error);

    // Empty spawn operator lists are rejected immediately.
    EXPECT_THROW(Source<float>::builder()
                     .with_spawn_operators(HostBuffer<SpawnOperator<float>> {}),
                 std::runtime_error);
}

TEST(Source, BuilderRejectsInvalidNumericConfiguration) {
    // Arrange: create the shared fluid dependency.
    const auto fluid = make_fluid();

    // Spacing must be positive.
    EXPECT_THROW(Source<float>::builder()
                     .with_units(HostBuffer<Unit<float>> { make_unit() })
                     .with_fluid(fluid)
                     .with_spawn_types(HostBuffer<SpawnType> { SpawnType::Surface })
                     .with_spawn_operator(make_spawn_operator())
                     .with_spacing(0.0f)
                     .build(),
                 std::runtime_error);

    // Tolerance must be finite.
    EXPECT_THROW(Source<float>::builder()
                     .with_units(HostBuffer<Unit<float>> { make_unit() })
                     .with_fluid(fluid)
                     .with_spawn_types(HostBuffer<SpawnType> { SpawnType::Surface })
                     .with_spawn_operator(make_spawn_operator())
                     .with_tolerance(std::numeric_limits<float>::infinity())
                     .build(),
                 std::runtime_error);
}

TEST(Source, MakeHostSharedBuildsSource) {
    // Arrange: create the shared fluid dependency.
    const auto fluid = make_fluid();

    // Act: build a source through host shared ownership.
    const auto source = Source<float>::builder()
                            .with_units(HostBuffer<Unit<float>> { make_unit() })
                            .with_fluid(fluid)
                            .with_spawn_types(HostBuffer<SpawnType> { SpawnType::Surface })
                            .with_spawn_operator(make_spawn_operator())
                            .make_host_shared();

    // Assert: the shared source exists and accepts updates.
    ASSERT_NE(source, nullptr);
    EXPECT_NO_THROW(source->update(0.1f));
}

TEST(Source, UpdateIgnoresNonPositiveDt) {
    // Arrange: create a valid source.
    const auto fluid = make_fluid();

    auto source = Source<float>::builder()
                      .with_units(HostBuffer<Unit<float>> { make_unit() })
                      .with_fluid(fluid)
                      .with_spawn_types(HostBuffer<SpawnType> { SpawnType::Surface })
                      .with_spawn_operator(make_spawn_operator())
                      .build();

    // Assert: non-positive dt is ignored without throwing.
    EXPECT_NO_THROW(source.update(0.0f));
}

TEST(Source, EmitIncreasesParticleCount) {
    // Arrange: create a volume source attached to an empty fluid.
    const auto fluid = make_fluid();

    auto source = Source<float>::builder()
                      .with_units(HostBuffer<Unit<float>> { make_unit() })
                      .with_fluid(fluid)
                      .with_spawn_types(HostBuffer<SpawnType> { SpawnType::Volume })
                      .with_spawn_operator(SpawnOperator<float>(SpawnType::Volume))
                      .with_spacing(1.0f)
                      .with_temperature(300.0f)
                      .build();

    // Act: emit particles into the fluid.
    source.emit();

    // Assert: emission populates only the active prefix within capacity.
    EXPECT_GT(fluid->particle_count(), 0u);
    EXPECT_LE(fluid->particle_count(), fluid->buffer_size());
}

TEST(Source, EmitMatchesExpectedVolumeParticleCountForBoxSpacing) {
    // Arrange: create enough capacity for every volume sample in a [-1, 1]^3 box.
    constexpr float spacing = 1.0f;
    const std::size_t samples_per_axis = box_axis_sample_count();
    const std::size_t expected_particle_count = samples_per_axis * samples_per_axis * samples_per_axis;
    const auto fluid = make_fluid(expected_particle_count);

    auto source = Source<float>::builder()
                      .with_units(HostBuffer<Unit<float>> { make_unit() })
                      .with_fluid(fluid)
                      .with_spawn_types(HostBuffer<SpawnType> { SpawnType::Volume })
                      .with_spawn_operator(SpawnOperator<float>(SpawnType::Volume))
                      .with_spacing(spacing)
                      .with_temperature(300.0f)
                      .build();

    // Act: emit once from the box volume.
    source.emit();

    // Assert: volume spawning emits one particle for each inclusive lattice sample.
    EXPECT_EQ(expected_particle_count, 27u);
    EXPECT_EQ(fluid->particle_count(), expected_particle_count);
}

TEST(Source, EmitMatchesExpectedSurfaceParticleCountForBoxSpacing) {
    // Arrange: create enough capacity for every surface sample in a [-1, 1]^3 box.
    constexpr float spacing = 1.0f;
    const std::size_t samples_per_axis = box_axis_sample_count();
    const std::size_t volume_sample_count = samples_per_axis * samples_per_axis * samples_per_axis;
    const std::size_t interior_sample_count = (samples_per_axis - 2u) * (samples_per_axis - 2u) * (samples_per_axis - 2u);
    const std::size_t expected_particle_count = volume_sample_count - interior_sample_count;
    const auto fluid = make_fluid(expected_particle_count);

    auto source = Source<float>::builder()
                      .with_units(HostBuffer<Unit<float>> { make_unit() })
                      .with_fluid(fluid)
                      .with_spawn_types(HostBuffer<SpawnType> { SpawnType::Surface })
                      .with_spawn_operator(SpawnOperator<float>(SpawnType::Surface))
                      .with_spacing(spacing)
                      .with_tolerance(0.0f)
                      .with_temperature(300.0f)
                      .build();

    // Act: emit once from the box surface.
    source.emit();

    // Assert: surface spawning emits every lattice sample except the single interior point.
    EXPECT_EQ(expected_particle_count, 26u);
    EXPECT_EQ(fluid->particle_count(), expected_particle_count);
}
