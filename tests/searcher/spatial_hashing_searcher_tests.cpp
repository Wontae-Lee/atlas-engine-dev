#include "../utilities/tests_utils.h"

#include <atlas/searcher/spatial_hashing_searcher.h>
#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

using namespace atlas;

TEST(SpatialHashingSearcher, BuilderRejectsNullDomain) {
    EXPECT_THROW(
        (void)atlas::SpatialHashingSearcher<float>::builder().build(),
        std::invalid_argument);
}

TEST(SpatialHashingSearcher, BuilderMakeHostSharedPreservesConfiguredProbeProperties) {
    const auto domain = test::make_domain_ptr<float>();

    auto searcher = atlas::SpatialHashingSearcher<float>::builder()
                        .with_domain(domain)
                        .make_host_shared();

    ASSERT_NE(searcher, nullptr);

    const auto probe = searcher->make_device_probe();

    EXPECT_TRUE(test::vec_near(probe.lower_corner, domain->lower_corner(), 1e-6f));
    EXPECT_EQ(probe.grid_size.x, domain->grid_size().x);
    EXPECT_EQ(probe.grid_size.y, domain->grid_size().y);
    EXPECT_EQ(probe.grid_size.z, domain->grid_size().z);
    EXPECT_FLOAT_EQ(probe.inv_h, domain->inverse_cell_size());
    EXPECT_FLOAT_EQ(probe.cell_size, domain->cell_size());
}

TEST(SpatialHashingSearcher, BuildPopulatesSortedIndicesAndCellRanges) {
    const auto domain = atlas::make_host_shared<system::Domain<float>>(
        Vector3<float>(0.0f, 0.0f, 0.0f),
        Vector3<float>(1.0f, 1.0f, 1.0f),
        0.5f);

    auto searcher = atlas::SpatialHashingSearcher<float>::builder()
                        .with_domain(domain)
                        .build();

    system::Fluid<float> particle_data(3);
    auto particle_probe = particle_data.make_device_probe();
    particle_probe.particle_count = 3;

    const std::array<Vector3<float>, 3> positions {
        Vector3<float>(0.10f, 0.10f, 0.10f),
        Vector3<float>(0.20f, 0.20f, 0.20f),
        Vector3<float>(0.80f, 0.80f, 0.80f)
    };

    atlas::copy_host_to_device(positions.data(), particle_probe.pos, positions.size());

    searcher.build(particle_probe);

    const auto probe = searcher.make_device_probe();

    const auto indices = test::copy_device_range(probe.indices, static_cast<std::size_t>(particle_probe.particle_count));
    const auto cell_start = test::copy_device_range(probe.cell_start, static_cast<std::size_t>(domain->number_of_cells()));
    const auto cell_end = test::copy_device_range(probe.cell_end, static_cast<std::size_t>(domain->number_of_cells()));

    ASSERT_EQ(indices.size(), 3u);
    EXPECT_EQ(indices[2], 2);
    EXPECT_TRUE((indices[0] == 0 && indices[1] == 1) || (indices[0] == 1 && indices[1] == 0));

    const auto first_cell = atlas::SpatialHashingSearcher<float>::linear_key(0, 0, 0, domain->grid_size());
    const auto last_cell = atlas::SpatialHashingSearcher<float>::linear_key(1, 1, 1, domain->grid_size());

    ASSERT_LT(static_cast<std::size_t>(first_cell), cell_start.size());
    ASSERT_LT(static_cast<std::size_t>(last_cell), cell_start.size());

    EXPECT_EQ(cell_start[first_cell], 0);
    EXPECT_EQ(cell_end[first_cell], 2);
    EXPECT_EQ(cell_start[last_cell], 2);
    EXPECT_EQ(cell_end[last_cell], 3);
}
