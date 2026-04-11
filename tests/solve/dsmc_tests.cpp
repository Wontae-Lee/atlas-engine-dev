#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(DsmcOperator, HardSphereSetsModelType) {
    const auto op = atlas::HardSphereDsmcOperator<double>(1.0, 7u);
    const auto dsmc_op = atlas::DsmcOperator<double>(op);

    EXPECT_EQ(dsmc_op.type, atlas::system::DsmcModelType::hard_sphere);
}

TEST(Dsmc, BuilderAndSetterStoreSolveMode) {
    const auto fluid = atlas::Fluid<double>::builder()
                           .add_species(
                               atlas::MatrialProperties<double>::builder()
                                   .with_type(atlas::MaterialType::Molecule)
                                   .with_mass(1.0)
                                   .with_collision_diameter(1.0)
                                   .build(),
                               1.0)
                           .with_buffer_size(2)
                           .make_host_shared();

    auto dsmc = atlas::Dsmc<double>::builder()
                    .with_fluid(fluid)
                    .with_solve_mode(atlas::DsmcSolveMode::ntc)
                    .build();

    EXPECT_EQ(dsmc.solve_mode(), atlas::DsmcSolveMode::ntc);

    dsmc.set_solve_mode(atlas::DsmcSolveMode::disjoint_pair);
    EXPECT_EQ(dsmc.solve_mode(), atlas::DsmcSolveMode::disjoint_pair);
}

TEST(Dsmc, SolveScattersPairedParticlesWhileConservingMomentum) {
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
                             .with_collision_diameter(1.0)
                             .build(),
                         1.0)
                     .with_buffer_size(4)
                     .make_host_shared();

    auto particle_probe = fluid->make_device_probe();
    particle_probe.particle_count = 2;

    fluid->positions()[0]   = Vector3<double>(0.25, 0.25, 0.25);
    fluid->positions()[1]   = Vector3<double>(0.75, 0.75, 0.75);
    fluid->velocities()[0]  = Vector3<double>(1.0, 0.0, 0.0);
    fluid->velocities()[1]  = Vector3<double>(-1.0, 0.0, 0.0);
    fluid->species_ids()[0] = 0;
    fluid->species_ids()[1] = 0;

    auto searcher = atlas::SpatialHashingSearcher<double>::builder()
                        .with_domain(atlas::make_host_shared<atlas::Domain<double>>(domain))
                        .build();
    searcher.build(particle_probe);

    auto dsmc = atlas::Dsmc<double>::builder()
                    .with_fluid(fluid)
                    .with_operator(atlas::DsmcOperator<double>(atlas::HardSphereDsmcOperator<double>(1.0, 42u)))
                    .build();

    auto domain_probe   = domain.make_device_probe();
    auto searcher_probe = searcher.make_device_probe();
    atlas::system::CodecDeviceProbe<double> codec_probe { atlas::system::CodecType::single, nullptr };

    const auto before_vel = test::copy_device_buffer(fluid->velocities());
    dsmc.solve(domain_probe, searcher_probe, particle_probe, codec_probe);
    const auto after_vel = test::copy_device_buffer(fluid->velocities());

    const auto before_momentum = before_vel[0] + before_vel[1];
    const auto after_momentum  = after_vel[0] + after_vel[1];

    EXPECT_TRUE(test::vec_near(before_momentum, after_momentum, 1e-12));
    EXPECT_TRUE(test::near(after_vel[0].length(), before_vel[0].length(), 1e-12));
    EXPECT_TRUE(test::near(after_vel[1].length(), before_vel[1].length(), 1e-12));
    EXPECT_FALSE(test::vec_near(after_vel[0], before_vel[0], 1e-12));
}

TEST(Dsmc, SolveAppliesDomainFieldForceToParticleVelocity) {
    auto domain = atlas::Domain<double>::builder()
                      .with_lower_corner(Vector3<double>(0.0, 0.0, 0.0))
                      .with_upper_corner(Vector3<double>(1.0, 1.0, 1.0))
                      .with_cell_size(1.0)
                      .build();
    domain.set_field_force(HostBuffer<Vector3<double>> { Vector3<double>(0.25, -0.5, 1.0) });

    const auto fluid = atlas::Fluid<double>::builder()
                           .add_species(
                               atlas::MatrialProperties<double>::builder()
                                   .with_type(atlas::MaterialType::Molecule)
                                   .with_mass(1.0)
                                   .with_collision_diameter(1.0)
                                   .build(),
                               1.0)
                           .with_buffer_size(2)
                           .make_host_shared();

    auto particle_probe = fluid->make_device_probe();
    particle_probe.particle_count = 1;

    fluid->positions()[0]   = Vector3<double>(0.25, 0.25, 0.25);
    fluid->velocities()[0]  = Vector3<double>(1.0, 2.0, 3.0);
    fluid->species_ids()[0] = 0;

    auto searcher = atlas::SpatialHashingSearcher<double>::builder()
                        .with_domain(atlas::make_host_shared<atlas::Domain<double>>(domain))
                        .build();
    searcher.build(particle_probe);

    auto dsmc = atlas::Dsmc<double>::builder()
                    .with_fluid(fluid)
                    .with_operator(atlas::DsmcOperator<double>(atlas::HardSphereDsmcOperator<double>(1.0, 42u)))
                    .build();

    auto domain_probe   = domain.make_device_probe();
    auto searcher_probe = searcher.make_device_probe();
    atlas::system::CodecDeviceProbe<double> codec_probe { atlas::system::CodecType::single, nullptr };

    dsmc.solve(domain_probe, searcher_probe, particle_probe, codec_probe);
    const auto after_vel = test::copy_device_buffer(fluid->velocities());

    EXPECT_TRUE(test::vec_near(after_vel[0], Vector3<double>(1.25, 1.5, 4.0), 1e-12));
}

TEST(Dsmc, SolveUsesDomainFieldTemperatureForVhsModel) {
    auto domain = atlas::Domain<double>::builder()
                      .with_lower_corner(Vector3<double>(0.0, 0.0, 0.0))
                      .with_upper_corner(Vector3<double>(1.0, 1.0, 1.0))
                      .with_cell_size(1.0)
                      .with_type(atlas::DomainType::isothermal)
                      .with_temperature(300.0)
                      .build();

    const auto fluid = atlas::Fluid<double>::builder()
                           .add_species(
                               atlas::MatrialProperties<double>::builder()
                                   .with_type(atlas::MaterialType::Molecule)
                                   .with_mass(1.0)
                                   .with_collision_diameter(1.0)
                                   .with_viscosity_index(1.0)
                                   .build(),
                               1.0)
                           .with_buffer_size(4)
                           .make_host_shared();

    auto particle_probe = fluid->make_device_probe();
    particle_probe.particle_count = 2;

    fluid->positions()[0]   = Vector3<double>(0.25, 0.25, 0.25);
    fluid->positions()[1]   = Vector3<double>(0.75, 0.75, 0.75);
    fluid->velocities()[0]  = Vector3<double>(0.5, 0.0, 0.0);
    fluid->velocities()[1]  = Vector3<double>(-0.5, 0.0, 0.0);
    fluid->species_ids()[0] = 0;
    fluid->species_ids()[1] = 0;

    auto searcher = atlas::SpatialHashingSearcher<double>::builder()
                        .with_domain(atlas::make_host_shared<atlas::Domain<double>>(domain))
                        .build();
    searcher.build(particle_probe);

    auto dsmc = atlas::Dsmc<double>::builder()
                    .with_fluid(fluid)
                    .with_operator(atlas::DsmcOperator<double>(atlas::VhsDsmcOperator<double>(0.0, 1.0e12, 7u)))
                    .build();

    auto domain_probe   = domain.make_device_probe();
    auto searcher_probe = searcher.make_device_probe();
    atlas::system::CodecDeviceProbe<double> codec_probe { atlas::system::CodecType::single, nullptr };

    const auto before_vel = test::copy_device_buffer(fluid->velocities());
    dsmc.solve(domain_probe, searcher_probe, particle_probe, codec_probe);
    const auto after_vel = test::copy_device_buffer(fluid->velocities());

    EXPECT_FALSE(test::vec_near(after_vel[0], before_vel[0], 1e-12));
    EXPECT_FALSE(test::vec_near(after_vel[1], before_vel[1], 1e-12));
}

TEST(Dsmc, NtcSolveScattersPairedParticlesWhileConservingMomentum) {
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
                                   .with_collision_diameter(1.0)
                                   .build(),
                               1.0)
                           .with_buffer_size(4)
                           .make_host_shared();

    auto particle_probe = fluid->make_device_probe();
    particle_probe.particle_count = 2;

    fluid->positions()[0]   = Vector3<double>(0.25, 0.25, 0.25);
    fluid->positions()[1]   = Vector3<double>(0.75, 0.75, 0.75);
    fluid->velocities()[0]  = Vector3<double>(1.0, 0.0, 0.0);
    fluid->velocities()[1]  = Vector3<double>(-1.0, 0.0, 0.0);
    fluid->species_ids()[0] = 0;
    fluid->species_ids()[1] = 0;

    auto searcher = atlas::SpatialHashingSearcher<double>::builder()
                        .with_domain(atlas::make_host_shared<atlas::Domain<double>>(domain))
                        .build();
    searcher.build(particle_probe);

    auto dsmc = atlas::Dsmc<double>::builder()
                    .with_fluid(fluid)
                    .with_operator(atlas::DsmcOperator<double>(atlas::HardSphereDsmcOperator<double>(1.0, 42u)))
                    .with_solve_mode(atlas::DsmcSolveMode::ntc)
                    .build();

    auto domain_probe   = domain.make_device_probe();
    auto searcher_probe = searcher.make_device_probe();
    atlas::system::CodecDeviceProbe<double> codec_probe { atlas::system::CodecType::single, nullptr };

    const auto before_vel = test::copy_device_buffer(fluid->velocities());
    dsmc.solve(domain_probe, searcher_probe, particle_probe, codec_probe);
    const auto after_vel = test::copy_device_buffer(fluid->velocities());

    const auto before_momentum = before_vel[0] + before_vel[1];
    const auto after_momentum  = after_vel[0] + after_vel[1];

    EXPECT_TRUE(test::vec_near(before_momentum, after_momentum, 1e-12));
    EXPECT_FALSE(test::vec_near(after_vel[0], before_vel[0], 1e-12));
}
