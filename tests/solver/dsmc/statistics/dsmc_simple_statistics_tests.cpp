#include <atlas/solver/dsmc/statistics/dsmc_simple_statistics.h>

#include "../dsmc_test_utils.h"

#include <atlas/memory/raw_pointer_cast.h>

#include <gtest/gtest.h>

namespace {

struct DsmcSimpleStatisticsFixture {
    atlas::DeviceBuffer<atlas::Vector3> velocity = atlas::DeviceBuffer<atlas::Vector3>(2);
    atlas::DeviceBuffer<float> number_particle = atlas::DeviceBuffer<float>(1);
    atlas::DeviceBuffer<float> max_relative_speed = atlas::DeviceBuffer<float>(1);
    atlas::DeviceBuffer<float> max_sigma_g = atlas::DeviceBuffer<float>(1);
    atlas::DeviceBuffer<float> collision_remainder = atlas::DeviceBuffer<float>(1);
    atlas::DeviceBuffer<int> collision_count = atlas::DeviceBuffer<int>(1);
    atlas::DeviceBuffer<int> indices = atlas::DeviceBuffer<int>(2);
    atlas::DeviceBuffer<int> cell_start = atlas::DeviceBuffer<int>(1);
    atlas::DeviceBuffer<int> cell_end = atlas::DeviceBuffer<int>(1);
    atlas::DsmcProbe probe {};

    DsmcSimpleStatisticsFixture() {
        velocity[0] = atlas::Vector3(1.0f, 0.0f, 0.0f);
        velocity[1] = atlas::Vector3(-1.0f, 0.0f, 0.0f);
        indices[0] = 0;
        indices[1] = 1;
        cell_start[0] = 0;
        cell_end[0] = 2;

        probe.velocity_ptr = atlas::raw_pointer_cast(velocity.data());
        probe.number_particle_ptr = atlas::raw_pointer_cast(number_particle.data());
        probe.max_relative_speed_ptr = atlas::raw_pointer_cast(max_relative_speed.data());
        probe.max_sigma_g_ptr = atlas::raw_pointer_cast(max_sigma_g.data());
        probe.collision_remainder_ptr = atlas::raw_pointer_cast(collision_remainder.data());
        probe.collision_count_ptr = atlas::raw_pointer_cast(collision_count.data());
        probe.indices_ptr = atlas::raw_pointer_cast(indices.data());
        probe.cell_start_ptr = atlas::raw_pointer_cast(cell_start.data());
        probe.cell_end_ptr = atlas::raw_pointer_cast(cell_end.data());
        probe.particle_count = 2;
        probe.cell_count = 1;
        probe.cell_volume = 1.0f;
        probe.statistical_weight = 1.0f;
    }
};

}

TEST(DsmcSimpleStatistics, MeasureHandlesEmptyProbe) {
    const atlas::DsmcSimpleStatistics statistics;
    const atlas::DsmcProbe probe;

    EXPECT_TRUE(statistics.measure(probe, nullptr, 0, 0.0f));
}

TEST(DsmcSimpleStatistics, MeasureUsesRelativeSpeedAsSimpleMajorant) {
    DsmcSimpleStatisticsFixture fixture;
    const atlas::DsmcSimpleStatistics statistics;

    ASSERT_TRUE(statistics.measure(fixture.probe, nullptr, 0, 1.0f));

    EXPECT_EQ(fixture.number_particle[0], 2.0f);
    EXPECT_NEAR(fixture.max_relative_speed[0], 2.0f, 1.0e-6f);
    EXPECT_NEAR(fixture.max_sigma_g[0], 2.0f, 1.0e-6f);
    EXPECT_EQ(fixture.collision_count[0], 2);
}

TEST(DsmcSimpleStatistics, MeasureRespectsAllocatedSolverFilter) {
    DsmcSimpleStatisticsFixture fixture;
    fixture.number_particle[0] = 5.0f;
    atlas::DeviceBuffer<int> owners(1);
    owners[0] = 9;
    const atlas::DsmcSimpleStatistics statistics;

    ASSERT_TRUE(statistics.measure(fixture.probe, &owners, 4, 1.0f));

    EXPECT_EQ(fixture.number_particle[0], 5.0f);
}
