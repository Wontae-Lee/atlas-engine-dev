#include <atlas/atlas.h>

#include <iostream>

namespace {

using T    = float;
using Vec3 = atlas::Vector3<T>;

namespace config {
    constexpr T kTemperature                                  = 300.0f;
    constexpr T kDt                                           = 2.5e-5f;
    constexpr std::size_t kBufferSize                         = 200000;
    constexpr T kNitrogenMolecularMass                        = 4.651734e-26f;
    constexpr T kNitrogenDiameter                             = 4.17e-10f;
    constexpr T kCellSize                                     = 0.25f;
    constexpr T kSourceSpacing                                = 0.18f;
    constexpr int kSteps                                      = 1000;
    constexpr T kCylinderRadius                               = 0.9f;
    constexpr T kCylinderHeight                               = 2.2f;
    constexpr T kRestitution                                  = 1.0f;
    constexpr T kTangentialMomentumAccommodation              = 1.0f;
    constexpr atlas::system::DiffuseSampling kDiffuseSampling = atlas::system::DiffuseSampling::CosineWeighted;
    constexpr atlas::system::DsmcKernelType kDsmcKernelType   = atlas::system::DsmcKernelType::hard_sphere;
    constexpr atlas::MeasureModeType kMeasureMode             = atlas::MeasureModeType::Field;
    const Vec3 kDomainMin(-6.0f, -2.5f, -1.2f);
    const Vec3 kDomainMax(6.0f, 2.5f, 1.2f);
    const Vec3 kSourceMin(-5.1f, -0.9f, -0.18f);
    const Vec3 kSourceMax(-4.0f, 0.9f, 0.18f);
    const Vec3 kCylinderCenter(0, 0, 0);
}

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
    atlas::HostBuffer<atlas::MaterialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    properties[0] = atlas::MaterialProperties<T>::builder()
                        .with_type(atlas::MaterialType::Molecule)
                        .with_mass(config::kNitrogenMolecularMass)
                        .with_molecular_mass(config::kNitrogenMolecularMass)
                        .with_species_id(0)
                        .with_reference_diameter(config::kNitrogenDiameter)
                        .build();

    generators[0] = atlas::fluid::MaxwellBoltzmannGenerator<T>::builder()
                        .with_temperature(config::kTemperature)
                        .with_molecular_mass(config::kNitrogenMolecularMass)
                        .with_bulk_velocity(Vec3(0, 0, 0))
                        .with_seed(42u)
                        .make_host_shared();

    return atlas::Fluid<T>::builder()
        .with_buffer_size(config::kBufferSize)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

atlas::GeometryHostPtr<T>
make_domain_geometry() {
    return atlas::geometry::Box<T>::builder()
        .with_lower_corner(config::kDomainMin)
        .with_upper_corner(config::kDomainMax)
        .make_host_shared();
}

atlas::GeometryHostPtr<T>
make_source_geometry() {
    return atlas::geometry::Box<T>::builder()
        .with_lower_corner(config::kSourceMin)
        .with_upper_corner(config::kSourceMax)
        .make_host_shared();
}

atlas::GeometryHostPtr<T>
make_cylinder_geometry() {
    return atlas::geometry::Cylinder<T>::builder()
        .with_center(config::kCylinderCenter)
        .with_radius(config::kCylinderRadius)
        .with_height(config::kCylinderHeight)
        .make_host_shared();
}

atlas::system::ColliderSurfaceInteraction<T>
make_collider_interaction() {
    return atlas::system::ColliderSurfaceInteraction<T>::builder()
        .with_diffuse_sampling(config::kDiffuseSampling)
        .with_restitution(config::kRestitution)
        .with_tangential_momentum_accommodation(config::kTangentialMomentumAccommodation)
        .with_temperature(config::kTemperature)
        .build();
}

}

int
main() {
    const auto fluid    = make_fluid();
    const auto universe = atlas::Universe<T>::builder()
                              .with_lower_corner(config::kDomainMin)
                              .with_upper_corner(config::kDomainMax)
                              .with_cell_size(config::kCellSize)
                              .make_host_shared();
    const auto searcher = atlas::SpatialHashingSearcher<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .make_host_shared();
    const auto dsmc_solver = atlas::DsmcSolver<T>::builder()
                                 .with_universe(universe)
                                 .with_fluid(fluid)
                                 .with_searcher(searcher)
                                 .with_kernel_type(config::kDsmcKernelType)
                                 .make_host_shared();
    const auto measurer = atlas::BoltzmanMeasurer<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .with_measure_mode(config::kMeasureMode)
                              .make_host_shared();
    const auto orchestrator = atlas::Orchestrator<T>::builder()
                                  .with_universe(universe)
                                  .with_fluid(fluid)
                                  .with_searcher(searcher)
                                  .with_measurer(measurer)
                                  .with_solver(dsmc_solver)
                                  .make_host_shared();
    const auto domain_unit   = make_unit(make_domain_geometry());
    const auto source_unit   = make_unit(make_source_geometry());
    const auto cylinder_unit = make_unit(make_cylinder_geometry());
    const auto source        = atlas::fluid::Source<T>::builder()
                            .with_units(atlas::HostBuffer<atlas::Unit<T>> { *source_unit })
                            .with_fluid(fluid)
                            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> {
                                atlas::fluid::SpawnType::Volume,
                            })
                            .with_spawn_operator(atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Volume))
                            .with_spacing(config::kSourceSpacing)
                            .with_temperature(config::kTemperature)
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
                                      make_collider_interaction(),
                                  })
                              .make_host_shared();
    const auto system = atlas::System<T>::builder()
                            .with_fluid(fluid)
                            .with_domain(universe)
                            .with_source(source)
                            .with_sink(sink)
                            .with_collider(collider)
                            .with_solver(orchestrator)
                            .with_dt(config::kDt)
                            .make_host_shared();

    for (int step = 0; step < config::kSteps; ++step) {
        system->update();
    }

    std::cout << "Cylinder rarefied gas completed " << config::kSteps << " steps.\n"
              << "Active particles: " << fluid->particle_count() << '\n';
    return 0;
}
