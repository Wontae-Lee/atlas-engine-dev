#include <atlas/atlas.h>

#include <iostream>

namespace {

using T    = float;
using Vec3 = atlas::Vector3<T>;

namespace config {
    constexpr T kDt                       = 1.0e-3f;
    constexpr std::size_t kBufferSize     = 120000;
    constexpr T kParticleMass             = 1.0f;
    constexpr T kRestDensity              = 1000.0f;
    constexpr T kPressureCoefficient      = 35.0f;
    constexpr T kDynamicViscosity         = 0.08f;
    constexpr T kSmoothingLength          = 0.28f;
    constexpr T kCellSize                 = 0.30f;
    constexpr T kSourceSpacing            = 0.16f;
    constexpr T kCylinderRadius           = 0.75f;
    constexpr T kCylinderHeight           = 2.2f;
    constexpr int kSteps                  = 1200;
    const Vec3 kDomainMin(-5.0f, -2.0f, -1.2f);
    const Vec3 kDomainMax(5.0f, 2.0f, 1.2f);
    const Vec3 kSourceMin(-4.5f, -0.7f, -0.2f);
    const Vec3 kSourceMax(-3.8f, 0.7f, 0.2f);
    const Vec3 kCylinderCenter(0, 0, 0);
    constexpr T kSourceBaseVelocity    = 2.4f;
    constexpr T kSourceVelocityJitter  = 0.25f;
    const Vec3 kGravity(0.0f, 0.0f, 0.0f);
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
                        .with_mass(config::kParticleMass)
                        .with_molecular_mass(config::kParticleMass)
                        .with_rest_density(config::kRestDensity)
                        .with_pressure_coefficient(config::kPressureCoefficient)
                        .with_dynamic_viscosity(config::kDynamicViscosity)
                        .with_smoothing_length(config::kSmoothingLength)
                        .with_species_id(0)
                        .build();

    generators[0] = atlas::fluid::JitteringOperator<T>::builder()
                        .with_base_value(config::kSourceBaseVelocity)
                        .with_jitter_radius(config::kSourceVelocityJitter)
                        .with_seed(17u)
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

}

int
main() {
    const auto fluid = make_fluid();
    const auto universe = atlas::Universe<T>::builder()
                              .with_lower_corner(config::kDomainMin)
                              .with_upper_corner(config::kDomainMax)
                              .with_cell_size(config::kCellSize)
                              .make_host_shared();
    const auto searcher = atlas::SpatialHashingSearcher<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .make_host_shared();
    const auto sph_solver = atlas::SphSolver<T>::builder()
                                .with_universe(universe)
                                .with_fluid(fluid)
                                .with_searcher(searcher)
                                .with_kernel_type(atlas::system::SphKernelType::cubic_spline)
                                .make_host_shared();
    const auto orchestrator = atlas::Orchestrator<T>::builder()
                                  .with_universe(universe)
                                  .with_fluid(fluid)
                                  .with_searcher(searcher)
                                  .with_gravity(config::kGravity)
                                  .with_solver(sph_solver)
                                  .make_host_shared();
    const auto domain_unit = make_unit(make_domain_geometry());
    const auto source_unit = make_unit(make_source_geometry());
    const auto cylinder_unit = make_unit(make_cylinder_geometry());
    const auto source = atlas::fluid::Source<T>::builder()
                            .with_units(atlas::HostBuffer<atlas::Unit<T>> { *source_unit })
                            .with_fluid(fluid)
                            .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> {
                                atlas::fluid::SpawnType::Volume,
                            })
                            .with_spawn_operator(atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Volume))
                            .with_spacing(config::kSourceSpacing)
                            .with_temperature(0.0f)
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

    std::cout << "Cylinder continum completed " << config::kSteps << " steps.\n"
              << "Active particles: " << fluid->particle_count() << '\n';
    return 0;
}
