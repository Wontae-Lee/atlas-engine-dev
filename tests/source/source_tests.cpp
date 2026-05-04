#include "../utilities/tests_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/geometry/box.h>
#include <atlas/source/source.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

atlas::FluidHostPtr<T>
make_fluid() {
    atlas::HostBuffer<atlas::MaterialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    properties[0] = atlas::MaterialProperties<T>::builder()
                        .with_type(atlas::MaterialType::Molecule)
                        .with_mass(4.651734e-26f)
                        .with_molecular_mass(4.651734e-26f)
                        .with_species_id(0)
                        .with_collision_diameter(4.17e-10f)
                        .build();

    generators[0] = atlas::fluid::MaxwellBoltzmannGenerator<T>::builder()
                        .with_temperature(300.0f)
                        .with_molecular_mass(4.651734e-26f)
                        .with_bulk_velocity(Vec3(0, 0, 0))
                        .with_seed(7u)
                        .make_host_shared();

    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(8)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

atlas::Unit<T>
make_unit() {
    const auto geometry = atlas::geometry::Box<T>::builder()
                              .with_lower_corner(Vec3(-1, -1, -1))
                              .with_upper_corner(Vec3(1, 1, 1))
                              .make_host_shared();

    const auto sync = atlas::physics::Sync<T>::builder()
                          .make_host_shared();

    return atlas::physics::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .build();
}

atlas::fluid::SpawnOperator<T>
make_spawn_operator() {
    return atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Surface);
}

} // namespace

TEST(Source, BuilderConstructsUsableSource) {
    const auto fluid = make_fluid();

    auto source = atlas::fluid::Source<T>::builder()
                      .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
                      .with_fluid(fluid)
                      .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> { atlas::fluid::SpawnType::Surface })
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

    EXPECT_THROW(
        atlas::fluid::Source<T>::builder()
            .with_fluid(fluid)
            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> { atlas::fluid::SpawnType::Surface })
            .with_spawn_operator(make_spawn_operator())
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::Source<T>::builder()
            .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> { atlas::fluid::SpawnType::Surface })
            .with_spawn_operator(make_spawn_operator())
            .build(),
        std::runtime_error);
}

TEST(Source, BuilderRejectsMismatchedSpawnConfigurationSizes) {
    const auto fluid = make_fluid();

    EXPECT_THROW(
        atlas::fluid::Source<T>::builder()
            .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit(), make_unit() })
            .with_fluid(fluid)
            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> { atlas::fluid::SpawnType::Surface, atlas::fluid::SpawnType::Volume, atlas::fluid::SpawnType::Surface })
            .with_spawn_operator(make_spawn_operator())
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::Source<T>::builder()
            .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit(), make_unit() })
            .with_fluid(fluid)
            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> { atlas::fluid::SpawnType::Surface })
            .with_spawn_operators(atlas::HostBuffer<atlas::fluid::SpawnOperator<T>> { make_spawn_operator(), make_spawn_operator(), make_spawn_operator() })
            .build(),
        std::runtime_error);
}

TEST(Source, BuilderRejectsInvalidImmediateInputs) {
    EXPECT_THROW(
        atlas::fluid::Source<T>::builder()
            .with_units(atlas::HostBuffer<atlas::Unit<T>> {}),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::Source<T>::builder()
            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> {}),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::Source<T>::builder()
            .with_spawn_operators(atlas::HostBuffer<atlas::fluid::SpawnOperator<T>> {}),
        std::runtime_error);
}

TEST(Source, BuilderRejectsInvalidNumericConfiguration) {
    const auto fluid = make_fluid();

    EXPECT_THROW(
        atlas::fluid::Source<T>::builder()
            .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
            .with_fluid(fluid)
            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> { atlas::fluid::SpawnType::Surface })
            .with_spawn_operator(make_spawn_operator())
            .with_spacing(0.0f)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::Source<T>::builder()
            .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
            .with_fluid(fluid)
            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> { atlas::fluid::SpawnType::Surface })
            .with_spawn_operator(make_spawn_operator())
            .with_tolerance(std::numeric_limits<T>::infinity())
            .build(),
        std::runtime_error);
}

TEST(Source, MakeHostSharedBuildsSource) {
    const auto fluid = make_fluid();

    const auto source = atlas::fluid::Source<T>::builder()
                            .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
                            .with_fluid(fluid)
                            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> { atlas::fluid::SpawnType::Surface })
                            .with_spawn_operator(make_spawn_operator())
                            .make_host_shared();

    ASSERT_NE(source, nullptr);
    EXPECT_NO_THROW(source->update(0.1f));
}

TEST(Source, UpdateIgnoresNonPositiveDt) {
    const auto fluid = make_fluid();

    auto source = atlas::fluid::Source<T>::builder()
                      .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
                      .with_fluid(fluid)
                      .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> { atlas::fluid::SpawnType::Surface })
                      .with_spawn_operator(make_spawn_operator())
                      .build();

    EXPECT_NO_THROW(source.update(0.0f));
}

TEST(Source, EmitIncreasesParticleCount) {
    const auto fluid = make_fluid();

    auto source = atlas::fluid::Source<T>::builder()
                      .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
                      .with_fluid(fluid)
                      .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> { atlas::fluid::SpawnType::Volume })
                      .with_spawn_operator(atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Volume))
                      .with_spacing(1.0f)
                      .with_temperature(300.0f)
                      .build();

    source.emit();

    EXPECT_GT(fluid->particle_count(), 0u);
    EXPECT_LE(fluid->particle_count(), fluid->buffer_size());
}
