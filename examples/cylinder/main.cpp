#include <atlas/atlas.h>

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/vizkit.h>
#endif

#include <iostream>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kTemperature = 300.0f;
constexpr T kDt = 2.5e-5f;
constexpr std::size_t kBufferSize = 200000;
constexpr T kNitrogenMolecularMass = 4.651734e-26f;
constexpr T kNitrogenDiameter = 4.17e-10f;

atlas::UnitHostPtr<T>
make_unit(const atlas::GeometryHostPtr<T>& geometry) {
    const auto sync = atlas::Sync<T>::builder()
                          .with_rigid_pose(Vec3(0, 0, 0), atlas::Quaternion<T>(1, 0, 0, 0))
                          .make_host_shared();

    return atlas::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .make_host_shared();
}

atlas::FluidHostPtr<T>
make_fluid() {
    atlas::HostBuffer<atlas::MatrialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    properties[0] = atlas::MatrialProperties<T>::builder()
                        .with_type(atlas::MaterialType::Molecule)
                        .with_mass(kNitrogenMolecularMass)
                        .with_molecular_mass(kNitrogenMolecularMass)
                        .with_species_id(0)
                        .with_collision_diameter(kNitrogenDiameter)
                        .build();

    generators[0] = atlas::fluid::MaxwellBoltzmannGenerator<T>::builder()
                        .with_temperature(kTemperature)
                        .with_molecular_mass(kNitrogenMolecularMass)
                        .with_bulk_velocity(Vec3(0, 0, 0))
                        .with_seed(42u)
                        .make_host_shared();

    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(kBufferSize)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

} // namespace

int
main() {
    const Vec3 domain_min(-6.0f, -2.5f, -1.2f);
    const Vec3 domain_max(6.0f, 2.5f, 1.2f);

    const auto fluid = make_fluid();

    const auto universe = atlas::Universe<T>::builder()
                              .with_lower_corner(domain_min)
                              .with_upper_corner(domain_max)
                              .with_cell_size(0.25f)
                              .make_host_shared();

    const auto searcher = atlas::SpatialHashingSearcher<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .make_host_shared();

    const auto dsmc_solver = atlas::DsmcNtcSolver<T>::builder()
                                 .with_universe(universe)
                                 .with_fluid(fluid)
                                 .with_searcher(searcher)
                                 .with_kernel_type(atlas::system::DsmcKernelType::hard_sphere)
                                 .with_collision_rate_scale(1.0f)
                                 .make_host_shared();

    const auto orchestrator = atlas::Orchestrator<T>::builder()
                                  .with_universe(universe)
                                  .with_fluid(fluid)
                                  .with_searcher(searcher)
                                  .with_solver(dsmc_solver)
                                  .make_host_shared();

    const auto domain_geometry = atlas::geometry::Box<T>::builder()
                                     .with_lower_corner(domain_min)
                                     .with_upper_corner(domain_max)
                                     .make_host_shared();

    const auto source_geometry = atlas::geometry::Box<T>::builder()
                                     .with_lower_corner(Vec3(-5.1f, -0.9f, -0.18f))
                                     .with_upper_corner(Vec3(-4.0f, 0.9f, 0.18f))
                                     .make_host_shared();

    const auto cylinder_geometry = atlas::geometry::Cylinder<T>::builder()
                                       .with_center(Vec3(0, 0, 0))
                                       .with_radius(0.9f)
                                       .with_height(2.2f)
                                       .make_host_shared();

    const auto domain_unit = make_unit(domain_geometry);
    const auto source_unit = make_unit(source_geometry);
    const auto cylinder_unit = make_unit(cylinder_geometry);

    const auto source = atlas::fluid::Source<T>::builder()
                            .with_units(atlas::HostBuffer<atlas::Unit<T>> { *source_unit })
                            .with_fluid(fluid)
                            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> {
                                atlas::fluid::SpawnType::Volume,
                            })
                            .with_spawn_operator(atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Volume))
                            .with_spacing(0.18f)
                            .with_temperature(kTemperature)
                            .make_host_shared();

    const auto sink = atlas::fluid::Sink<T>::builder()
                          .with_units(atlas::HostBuffer<atlas::Unit<T>> { *domain_unit })
                          .with_fluid(fluid)
                          .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> {
                              atlas::fluid::DespawnType::Volume,
                          })
                          .with_despawn_operator(atlas::fluid::DespawnOperator<T>(atlas::fluid::DespawnType::Volume))
                          .with_flip(true)
                          .make_host_shared();

    const auto collider = atlas::Collider<T>::builder()
                              .with_fluid(fluid)
                              .with_units(atlas::HostBuffer<atlas::Unit<T>> { *cylinder_unit })
                              .with_surface_interactions(
                                  atlas::HostBuffer<atlas::system::ColliderSurfaceInteraction<T>> {
                                      atlas::system::ColliderSurfaceInteraction<T>::builder()
                                          .with_diffuse_sampling(atlas::system::DiffuseSampling::CosineWeighted)
                                          .with_restitution(1.0f)
                                          .with_tangential_momentum_accommodation(1.f)
                                          .with_temperature(kTemperature)
                                          .build(),
                                  })
                              .make_host_shared();

    const auto system = atlas::System<T>::builder()
                            .with_fluid(fluid)
                            .with_domain(universe)
                            .with_source(source)
                            .with_sink(sink)
                            .with_collider(collider)
                            .with_solver(orchestrator)
                            .with_dt(kDt)
                            .make_host_shared();

#ifdef ATLAS_ENABLE_VIZKIT
    atlas::vizkit::Viewer<T> viewer = atlas::vizkit::Viewer<T>::builder()
                                          .with_system(system)
                                          .with_title("Atlas DSMC Nitrogen Cylinder")
                                          .with_size(1440, 900)
                                          .build();

    viewer.camera().fit_bounds(domain_min, domain_max);

    viewer.add_layer(
        atlas::vizkit::ParticleLayer<T>::builder()
            .with_system(system)
            .with_color(atlas::Vector4<T>(0.10f, 0.74f, 0.92f, 0.80f))
            .make_shared());

    viewer.add_layer(
        atlas::vizkit::BoxLayer<T>::builder()
            .with_unit(domain_unit)
            .make_shared());

    viewer.add_layer(
        atlas::vizkit::CylinderLayer<T>::builder()
            .with_unit(cylinder_unit)
            .with_slices(72)
            .make_shared());

    std::cout
        << "Nitrogen DSMC example: 300 K, zero bulk velocity, inlet-only source, outlet sinks, centered cylinder.\n";

    return viewer.run();
#else
    for (int step = 0; step < 1000; ++step) {
        system->update();
    }

    std::cout
        << "Nitrogen DSMC example ran headlessly for 1000 steps.\n"
        << "Active particles: " << fluid->particle_count() << '\n';
    return 0;
#endif
}
