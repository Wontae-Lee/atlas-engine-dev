#include <atlas/atlas.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace {

atlas::SyncHostPtr
identity_sync() {
    return atlas::Sync::builder()
        .with_rigid_pose(atlas::Float3(0.0f), atlas::Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
        .make_host_shared();
}

atlas::Unit
box_unit(const atlas::Float3& lower, const atlas::Float3& upper) {
    const atlas::Geometry geometry(
        atlas::Box::builder().with_lower_corner(lower).with_upper_corner(upper).build());
    return atlas::Unit::builder()
        .with_geometry(geometry)
        .with_sync(identity_sync())
        .build();
}

}

int
main(int argc, char** argv) {
    const atlas::Float3 domain_lower(-1.0f, -0.75f, -0.5f);
    const atlas::Float3 domain_upper(1.5f, 0.75f, 0.5f);
    const float         cell_size        = 0.1f;
    const float         dt               = 5.0e-5f;
    const std::size_t   steps            = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 40;
    const float         freestream_speed = 500.0f;
    const float         temperature      = 300.0f;
    const float         cylinder_radius  = 0.2f;

    auto materials = atlas::MaterialDictionary::builder()
                         .with_material(atlas::Material(atlas::Molecule(4.65e-26f,
                                                                        0.0f,
                                                                        0.0f,
                                                                        0.0f,
                                                                        4.17e-10f,
                                                                        273.0f,
                                                                        0.74f,
                                                                        1.0f)))
                         .make_host_shared();

    auto fluid = atlas::Fluid::builder()
                     .with_buffer_size(20'000)
                     .with_particle_count(0)
                     .with_statistical_weight(3.236e16f)
                     .with_materials(materials)
                     .make_host_unique();

    auto universe = atlas::Universe::builder()
                        .with_lower_corner(domain_lower)
                        .with_upper_corner(domain_upper)
                        .with_cell_size(cell_size)
                        .make_host_unique();

    const int cell_count = universe->cell_count();

    auto inflow = atlas::make_host_shared<atlas::Source>(
        atlas::Source(atlas::VolumeSource::builder()
                          .with_unit(box_unit(atlas::Float3(-0.8f, -0.5f, -0.35f),
                                              atlas::Float3(-0.7f, 0.5f, 0.35f)))
                          .with_spacing(cell_size)
                          .build()));

    auto freestream = atlas::make_host_shared<atlas::Generator>(
        atlas::Generator(atlas::MaxwellBoltzmannGenerator::builder()
                             .with_species_ratios({ 1.0f })
                             .with_species_numbers({ 0.0f })
                             .with_material_dictionary(*materials)
                             .with_temperature(temperature)
                             .with_bulk_velocity(atlas::Float3(freestream_speed, 0.0f, 0.0f))
                             .with_seed(20260921u)
                             .build()));

    const atlas::Geometry cylinder_geometry(
        atlas::Cylinder::builder()
            .with_center(atlas::Float3(0.0f))
            .with_radius(cylinder_radius)
            .with_height(1.0f)
            .with_open(true)
            .build());

    const atlas::Collider cylinder(
        atlas::IsothermalCollider::builder()
            .with_unit(atlas::Unit::builder()
                           .with_geometry(cylinder_geometry)
                           .with_sync(identity_sync())
                           .build())
            .with_momentum_accommodation_coefficient(1.0f)
            .with_restitution(1.0f)
            .build());

    const std::array<std::pair<atlas::Float3, atlas::Float3>, 6> sink_bounds = {
        std::pair { atlas::Float3(-1.0f, -0.75f, -0.5f), atlas::Float3(-0.9f, 0.75f, 0.5f) },
        std::pair { atlas::Float3(1.35f, -0.75f, -0.5f), atlas::Float3(1.5f, 0.75f, 0.5f) },
        std::pair { atlas::Float3(-1.0f, -0.75f, -0.5f), atlas::Float3(1.5f, -0.6f, 0.5f) },
        std::pair { atlas::Float3(-1.0f, 0.6f, -0.5f), atlas::Float3(1.5f, 0.75f, 0.5f) },
        std::pair { atlas::Float3(-1.0f, -0.75f, -0.5f), atlas::Float3(1.5f, 0.75f, -0.4f) },
        std::pair { atlas::Float3(-1.0f, -0.75f, 0.4f), atlas::Float3(1.5f, 0.75f, 0.5f) },
    };

    auto builder = atlas::System::builder();
    builder.with_fluid(std::move(fluid))
        .with_universe(std::move(universe))
        .with_dt(dt)
        .with_solver(atlas::DsmcSolver::builder().make_host_shared())
        .with_emitter(inflow, freestream)
        .with_collider(cylinder);

    for (const auto& [lower, upper] : sink_bounds) {
        builder.with_sink(atlas::Sink(
            atlas::VolumeSink::builder().with_unit(box_unit(lower, upper)).build()));
    }

    auto system = builder.build();

    std::printf("cylinder flow: cells=%d dt=%.1es steps=%zu\n", cell_count, system.dt(), steps);

    for (std::size_t step = 0; step < steps; ++step) {
        system.update();
        if (system.step() % 10 == 0 || system.step() == steps) {
            std::printf("step=%3zu particles=%5zu\n",
                        system.step(),
                        system.fluid()->particle_count());
        }
    }

    const std::size_t particle_count = system.fluid()->particle_count();
    const auto& position_data = system.fluid()->state<atlas::FluidPositionState>()->data();
    const auto& velocity_data = system.fluid()->state<atlas::FluidVelocityState>()->data();
    const atlas::HostBuffer<atlas::Float3> positions(position_data.begin(),
                                                      position_data.begin() + particle_count);
    const atlas::HostBuffer<atlas::Float3> velocities(velocity_data.begin(),
                                                       velocity_data.begin() + particle_count);

    std::size_t downstream = 0;
    double      mean_streamwise_velocity = 0.0;
    for (std::size_t index = 0; index < particle_count; ++index) {
        if (positions[index].x > cylinder_radius) ++downstream;
        mean_streamwise_velocity += velocities[index].x;
    }
    if (particle_count > 0) {
        mean_streamwise_velocity /= static_cast<double>(particle_count);
    }

    std::printf("completed: steps=%zu particles=%zu downstream=%zu mean_u=%.1fm/s\n",
                system.step(),
                particle_count,
                downstream,
                mean_streamwise_velocity);
    return 0;
}
