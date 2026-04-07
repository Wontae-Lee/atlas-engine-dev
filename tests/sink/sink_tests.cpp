#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Sink, VolumeSinkRemovesInteriorActiveParticlesAndCompactsProbe) {
    atlas::system::System<double> sim_system(5);
    auto& particle_probe          = sim_system.particle_probe();
    particle_probe.particle_count = 3;

    particle_probe.pos[0]     = Vector3<double>(0.0, 0.0, 0.0);
    particle_probe.pos[1]     = Vector3<double>(2.0, 0.0, 0.0);
    particle_probe.pos[2]     = Vector3<double>(0.5, 0.0, 0.0);
    particle_probe.pos[3]     = Vector3<double>(3.0, 0.0, 0.0);
    particle_probe.pos[4]     = Vector3<double>(0.0, 0.0, 0.0);
    particle_probe.vel[0]     = Vector3<double>(1.0, 0.0, 0.0);
    particle_probe.vel[1]     = Vector3<double>(2.0, 0.0, 0.0);
    particle_probe.vel[2]     = Vector3<double>(3.0, 0.0, 0.0);
    particle_probe.vel[3]     = Vector3<double>(4.0, 0.0, 0.0);
    particle_probe.vel[4]     = Vector3<double>(5.0, 0.0, 0.0);
    particle_probe.temperature[0] = 300.0;
    particle_probe.temperature[1] = 301.0;
    particle_probe.temperature[2] = 302.0;
    particle_probe.temperature[3] = 303.0;
    particle_probe.temperature[4] = 304.0;
    particle_probe.species[0] = 10;
    particle_probe.species[1] = 11;
    particle_probe.species[2] = 12;
    particle_probe.species[3] = 13;
    particle_probe.species[4] = 14;
    particle_probe.acitve[0]  = 1;
    particle_probe.acitve[1]  = 1;
    particle_probe.acitve[2]  = 1;
    particle_probe.acitve[3]  = 0;
    particle_probe.acitve[4]  = 0;

    const auto geometry = geometry::Box<double>::builder()
                              .with_lower_corner(Vector3<double>(-1.0, -1.0, -1.0))
                              .with_upper_corner(Vector3<double>(1.0, 1.0, 1.0))
                              .make_host_shared();
    const auto sync = atlas::system::Sync<double>::builder().make_host_shared();
    const auto unit = atlas::system::Unit<double>::builder()
                          .with_geometry(geometry)
                          .with_sync(sync)
                          .build();

    auto sink = atlas::system::Sink<double>::builder()
                    .with_unit(unit)
                    .with_despawn_type(atlas::system::DespawnType::Volume)
                    .with_tolerance(0.0)
                    .build();

    sink.sink(particle_probe);

    EXPECT_EQ(particle_probe.particle_count, 1);
    EXPECT_TRUE(test::vec_near(particle_probe.pos[0], Vector3<double>(2.0, 0.0, 0.0), 1e-12));
    EXPECT_TRUE(test::vec_near(particle_probe.vel[0], Vector3<double>(2.0, 0.0, 0.0), 1e-12));
    EXPECT_DOUBLE_EQ(particle_probe.temperature[0], 301.0);
    EXPECT_EQ(particle_probe.species[0], std::size_t(11));
    EXPECT_EQ(particle_probe.acitve[0], 1);
}

TEST(Sink, FlipInvertsVolumeDespawnClassification) {
    atlas::system::System<double> sim_system(5);
    auto& particle_probe          = sim_system.particle_probe();
    particle_probe.particle_count = 3;

    particle_probe.pos[0]     = Vector3<double>(0.0, 0.0, 0.0);
    particle_probe.pos[1]     = Vector3<double>(2.0, 0.0, 0.0);
    particle_probe.pos[2]     = Vector3<double>(0.5, 0.0, 0.0);
    particle_probe.vel[0]     = Vector3<double>(1.0, 0.0, 0.0);
    particle_probe.vel[1]     = Vector3<double>(2.0, 0.0, 0.0);
    particle_probe.vel[2]     = Vector3<double>(3.0, 0.0, 0.0);
    particle_probe.temperature[0] = 300.0;
    particle_probe.temperature[1] = 301.0;
    particle_probe.temperature[2] = 302.0;
    particle_probe.species[0] = 10;
    particle_probe.species[1] = 11;
    particle_probe.species[2] = 12;
    particle_probe.acitve[0]  = 1;
    particle_probe.acitve[1]  = 1;
    particle_probe.acitve[2]  = 1;

    const auto geometry = geometry::Box<double>::builder()
                              .with_lower_corner(Vector3<double>(-1.0, -1.0, -1.0))
                              .with_upper_corner(Vector3<double>(1.0, 1.0, 1.0))
                              .make_host_shared();
    const auto sync = atlas::system::Sync<double>::builder().make_host_shared();
    const auto unit = atlas::system::Unit<double>::builder()
                          .with_geometry(geometry)
                          .with_sync(sync)
                          .build();

    auto sink = atlas::system::Sink<double>::builder()
                    .with_unit(unit)
                    .with_despawn_type(atlas::system::DespawnType::Volume)
                    .with_flip(true)
                    .with_tolerance(0.0)
                    .build();

    sink.sink(particle_probe);

    EXPECT_EQ(particle_probe.particle_count, 2);
    EXPECT_TRUE(test::vec_near(particle_probe.pos[0], Vector3<double>(0.0, 0.0, 0.0), 1e-12));
    EXPECT_TRUE(test::vec_near(particle_probe.pos[1], Vector3<double>(0.5, 0.0, 0.0), 1e-12));
    EXPECT_DOUBLE_EQ(particle_probe.temperature[0], 300.0);
    EXPECT_DOUBLE_EQ(particle_probe.temperature[1], 302.0);
    EXPECT_EQ(particle_probe.species[0], std::size_t(10));
    EXPECT_EQ(particle_probe.species[1], std::size_t(12));
}
