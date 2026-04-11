#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(SphOperator, SpikySetsModelType) {
    const auto op = atlas::SpikySphOperator<double>(1.0);
    const auto sph_op = atlas::SphOperator<double>(op);

    EXPECT_EQ(sph_op.type, atlas::system::SphModelType::spiky);
}

TEST(Sph, SolveAppliesPairwiseVelocityCorrection) {
    auto domain = atlas::Domain<double>::builder()
                      .with_lower_corner(Vector3<double>(0.0, 0.0, 0.0))
                      .with_upper_corner(Vector3<double>(1.0, 1.0, 1.0))
                      .with_cell_size(1.0)
                      .build();

    const auto fluid = atlas::Fluid<double>::builder()
                           .add_species(
                               atlas::MatrialProperties<double>::builder()
                                   .with_type(atlas::MaterialType::Molecule)
                                   .with_mass(1.0)
                                   .with_rest_density(0.1)
                                   .with_pressure_coefficient(1.0)
                                   .with_dynamic_viscosity(0.5)
                                   .with_smoothing_length(1.0)
                                   .build(),
                               1.0)
                           .with_buffer_size(4)
                           .make_host_shared();

    auto particle_probe = fluid->make_device_probe();
    particle_probe.particle_count = 2;

    fluid->positions()[0]   = Vector3<double>(0.25, 0.25, 0.25);
    fluid->positions()[1]   = Vector3<double>(0.50, 0.25, 0.25);
    fluid->velocities()[0]  = Vector3<double>(0.0, 0.0, 0.0);
    fluid->velocities()[1]  = Vector3<double>(1.0, 0.0, 0.0);
    fluid->species_ids()[0] = 0;
    fluid->species_ids()[1] = 0;

    auto searcher = atlas::SpatialHashingSearcher<double>::builder()
                        .with_domain(atlas::make_host_shared<atlas::Domain<double>>(domain))
                        .build();
    searcher.build(particle_probe);

    auto sph = atlas::Sph<double>::builder()
                   .with_fluid(fluid)
                   .with_operator(atlas::SphOperator<double>(atlas::SpikySphOperator<double>(1.0)))
                   .build();

    auto domain_probe   = domain.make_device_probe();
    auto searcher_probe = searcher.make_device_probe();
    atlas::system::CodecDeviceProbe<double> codec_probe { atlas::system::CodecType::single, nullptr };

    const auto before_vel = test::copy_device_buffer(fluid->velocities());
    sph.solve(domain_probe, searcher_probe, particle_probe, codec_probe);
    const auto after_vel = test::copy_device_buffer(fluid->velocities());

    EXPECT_FALSE(test::vec_near(before_vel[0], after_vel[0], 1e-12));
    EXPECT_FALSE(test::vec_near(before_vel[1], after_vel[1], 1e-12));
}

TEST(Sph, SolveAppliesDomainFieldForceToParticleVelocity) {
    auto domain = atlas::Domain<double>::builder()
                      .with_lower_corner(Vector3<double>(0.0, 0.0, 0.0))
                      .with_upper_corner(Vector3<double>(1.0, 1.0, 1.0))
                      .with_cell_size(1.0)
                      .build();
    domain.set_field_force(HostBuffer<Vector3<double>> { Vector3<double>(0.1, 0.2, 0.3) });

    const auto fluid = atlas::Fluid<double>::builder()
                           .add_species(
                               atlas::MatrialProperties<double>::builder()
                                   .with_type(atlas::MaterialType::Molecule)
                                   .with_mass(1.0)
                                   .with_rest_density(0.1)
                                   .with_pressure_coefficient(1.0)
                                   .with_dynamic_viscosity(0.5)
                                   .with_smoothing_length(1.0)
                                   .build(),
                               1.0)
                           .with_buffer_size(2)
                           .make_host_shared();

    auto particle_probe = fluid->make_device_probe();
    particle_probe.particle_count = 1;

    fluid->positions()[0]   = Vector3<double>(0.25, 0.25, 0.25);
    fluid->velocities()[0]  = Vector3<double>(1.0, 1.0, 1.0);
    fluid->species_ids()[0] = 0;
    fluid->temperatures()[0] = 2.0;

    auto searcher = atlas::SpatialHashingSearcher<double>::builder()
                        .with_domain(atlas::make_host_shared<atlas::Domain<double>>(domain))
                        .build();
    searcher.build(particle_probe);

    auto sph = atlas::Sph<double>::builder()
                   .with_fluid(fluid)
                   .with_operator(atlas::SphOperator<double>(atlas::SpikySphOperator<double>(1.0)))
                   .build();

    auto domain_probe   = domain.make_device_probe();
    auto searcher_probe = searcher.make_device_probe();
    atlas::system::CodecDeviceProbe<double> codec_probe { atlas::system::CodecType::single, nullptr };

    sph.solve(domain_probe, searcher_probe, particle_probe, codec_probe);
    const auto after_vel = test::copy_device_buffer(fluid->velocities());

    EXPECT_TRUE(test::vec_near(after_vel[0], Vector3<double>(1.1, 1.2, 1.3), 1e-12));
}

TEST(Sph, SolveUsesParticleTemperatureInInteraction) {
    auto domain = atlas::Domain<double>::builder()
                      .with_lower_corner(Vector3<double>(0.0, 0.0, 0.0))
                      .with_upper_corner(Vector3<double>(1.0, 1.0, 1.0))
                      .with_cell_size(1.0)
                      .build();

    const auto make_fluid = []() {
        return atlas::Fluid<double>::builder()
            .add_species(
                atlas::MatrialProperties<double>::builder()
                    .with_type(atlas::MaterialType::Molecule)
                    .with_mass(1.0)
                    .with_rest_density(0.1)
                    .with_pressure_coefficient(1.0)
                    .with_dynamic_viscosity(0.5)
                    .with_smoothing_length(1.0)
                    .build(),
                1.0)
            .with_buffer_size(4)
            .make_host_shared();
    };

    const auto run_with_temperature = [&](const double temperature) {
        auto local_domain = atlas::Domain<double>::builder()
                                .with_lower_corner(Vector3<double>(0.0, 0.0, 0.0))
                                .with_upper_corner(Vector3<double>(1.0, 1.0, 1.0))
                                .with_cell_size(1.0)
                                .build();
        const auto fluid = make_fluid();
        auto particle_probe = fluid->make_device_probe();
        particle_probe.particle_count = 2;

        fluid->positions()[0]   = Vector3<double>(0.25, 0.25, 0.25);
        fluid->positions()[1]   = Vector3<double>(0.50, 0.25, 0.25);
        fluid->velocities()[0]  = Vector3<double>(0.0, 0.0, 0.0);
        fluid->velocities()[1]  = Vector3<double>(1.0, 0.0, 0.0);
        fluid->species_ids()[0] = 0;
        fluid->species_ids()[1] = 0;
        fluid->temperatures()[0] = temperature;
        fluid->temperatures()[1] = temperature;

        auto searcher = atlas::SpatialHashingSearcher<double>::builder()
                            .with_domain(atlas::make_host_shared<atlas::Domain<double>>(local_domain))
                            .build();
        searcher.build(particle_probe);

        auto sph = atlas::Sph<double>::builder()
                       .with_fluid(fluid)
                       .with_operator(atlas::SphOperator<double>(atlas::SpikySphOperator<double>(1.0)))
                       .build();

        auto domain_probe   = local_domain.make_device_probe();
        auto searcher_probe = searcher.make_device_probe();
        atlas::system::CodecDeviceProbe<double> codec_probe { atlas::system::CodecType::single, nullptr };

        const auto before_vel = test::copy_device_buffer(fluid->velocities());
        sph.solve(domain_probe, searcher_probe, particle_probe, codec_probe);
        const auto after_vel = test::copy_device_buffer(fluid->velocities());

        return (after_vel[0] - before_vel[0]).length();
    };

    const double low_temperature_delta = run_with_temperature(1.0);
    const double high_temperature_delta = run_with_temperature(3.0);

    EXPECT_GT(high_temperature_delta, low_temperature_delta);
}
