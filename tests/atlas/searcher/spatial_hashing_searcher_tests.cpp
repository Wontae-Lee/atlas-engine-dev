#include <atlas/searcher/spatial_hashing_searcher.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/searcher/spatial_hashing_searcher_view.h>
#include <atlas/universe/universe_state.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using atlas::copy_device_to_host;
using atlas::copy_host_to_device;
using atlas::DeviceBuffer;
using atlas::ExecutionPolicy;
using atlas::Float3;
using atlas::FluidPositionState;
using atlas::Int3;
using atlas::raw_pointer_cast;
using atlas::SpatialHashingSearcher;
using atlas::SpatialHashingSearcherView;
using atlas::UniverseNumberParticleState;

/** The view is captured by value inside a device lambda, so it must stay trivially copyable. */
static_assert(std::is_trivially_copyable_v<SpatialHashingSearcherView>,
              "SpatialHashingSearcherView must be trivially copyable for device capture");

/** Stage a host point set into a device-resident position column. */
FluidPositionState
make_positions(const std::vector<Float3>& points) {
    DeviceBuffer<Float3> buffer(points.size());
    copy_host_to_device(points.data(), buffer, points.size());
    return FluidPositionState(std::move(buffer));
}

/** Copy @p count ints from a raw device pointer into a host vector for inspection. */
std::vector<int>
to_host(const int* device_ptr, const std::size_t count) {
    std::vector<int> host(count);
    copy_device_to_host(device_ptr, host.data(), count);
    return host;
}

/** Build the same grid the tests reuse, through the validating fluent builder. */
SpatialHashingSearcher
make_searcher(const Float3& lower, const float cell_size, const Int3& grid) {
    return SpatialHashingSearcher::builder()
        .with_lower_corner(lower)
        .with_cell_size(cell_size)
        .with_grid_size(grid)
        .build();
}

/** Host histogram of per-cell particle counts, computed with the engine's own hash. */
std::vector<int>
reference_histogram(const std::vector<Float3>& points,
                    const Float3& lower,
                    const float inverse_cell_size,
                    const Int3& grid) {
    std::vector<int> histogram(static_cast<std::size_t>(grid.x * grid.y * grid.z), 0);
    for (const Float3& p : points) {
        const Int3 cell            = SpatialHashingSearcher::cell_for(p, lower, inverse_cell_size, grid);
        const std::uint32_t key    = SpatialHashingSearcher::linear_key(cell, grid);
        histogram[static_cast<std::size_t>(key)] += 1;
    }
    return histogram;
}

/**
 * @brief Neighbour query built on the view: which particles fall within @p radius of @p query.
 *
 * A single device thread walks every grid cell overlapping the query sphere's AABB,
 * enumerates each cell's particles through cell_start/cell_end/sorted_index, and flags
 * every particle inside the radius. The flag array (one int per particle) is copied back
 * so the test can compare it to a brute-force reference.
 */
std::vector<int>
query_neighbor_flags(const SpatialHashingSearcher& searcher,
                     const FluidPositionState& positions,
                     const int particle_count,
                     const Float3& query,
                     const float radius) {
    const SpatialHashingSearcherView view = searcher.view();

    const Float3* position_ptr = raw_pointer_cast(positions.data().data());

    DeviceBuffer<int> flags(static_cast<std::size_t>(particle_count), 0);
    int* flags_ptr = raw_pointer_cast(flags.data());

    const Float3 lower = searcher.lower_corner();
    const float inv    = searcher.inverse_cell_size();
    const Int3 grid    = searcher.grid_size();
    const float r2     = radius * radius;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        1,
        [=] ATLAS_ALL_DEVICE(const int) {
            const Float3 radius_vector(radius);
            const Int3 low  = SpatialHashingSearcher::cell_for(query - radius_vector, lower, inv, grid);
            const Int3 high = SpatialHashingSearcher::cell_for(query + radius_vector, lower, inv, grid);

            for (int iz = low.z; iz <= high.z; ++iz) {
                for (int iy = low.y; iy <= high.y; ++iy) {
                    for (int ix = low.x; ix <= high.x; ++ix) {
                        const std::uint32_t key = SpatialHashingSearcher::linear_key(ix, iy, iz, grid);
                        const int start         = view.cell_start[key];
                        if (start < 0) continue;
                        const int end = view.cell_end[key];
                        for (int slot = start; slot < end; ++slot) {
                            const int particle = view.sorted_index[slot];
                            if ((position_ptr[particle] - query).length_squared() <= r2) {
                                flags_ptr[particle] = 1;
                            }
                        }
                    }
                }
            }
        });

    return to_host(flags_ptr, static_cast<std::size_t>(particle_count));
}

/** Brute-force O(N^2) reference: flag every particle within @p radius of @p query. */
std::vector<int>
brute_force_flags(const std::vector<Float3>& points, const Float3& query, const float radius) {
    const float r2 = radius * radius;
    std::vector<int> flags(points.size(), 0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        if ((points[i] - query).length_squared() <= r2) {
            flags[i] = 1;
        }
    }
    return flags;
}

} // namespace

TEST(SpatialHashingSearcher, DefaultConstructionIsUnitGrid) {
    const SpatialHashingSearcher searcher;

    EXPECT_FLOAT_EQ(searcher.cell_size(), 1.0f);
    EXPECT_FLOAT_EQ(searcher.inverse_cell_size(), 1.0f);
    EXPECT_EQ(searcher.cell_count(), 1);
    EXPECT_EQ(searcher.grid_size(), Int3(1, 1, 1));
    EXPECT_EQ(searcher.lower_corner(), Float3(0.0f, 0.0f, 0.0f));
}

TEST(SpatialHashingSearcher, DefaultConstructionBacksTheCellsItReports) {
    const SpatialHashingSearcher searcher;
    const auto cells = static_cast<std::size_t>(searcher.cell_count());

    // The default constructor must size the ranges to cell_count() like the explicit one
    // does; reporting a cell it does not back would send a consumer past the buffer's end.
    ASSERT_NE(searcher.cell_start(), nullptr);
    ASSERT_NE(searcher.cell_end(), nullptr);

    const std::vector<int> start = to_host(searcher.cell_start(), cells);
    const std::vector<int> end   = to_host(searcher.cell_end(), cells);

    EXPECT_EQ(start[0], -1);
    EXPECT_EQ(end[0], -1);
}

TEST(SpatialHashingSearcher, ConstructedGeometryCachesDerivedValues) {
    const SpatialHashingSearcher searcher = make_searcher(Float3(-1.0f, 2.0f, 3.0f), 0.5f, Int3(4, 3, 2));

    EXPECT_FLOAT_EQ(searcher.cell_size(), 0.5f);
    EXPECT_FLOAT_EQ(searcher.inverse_cell_size(), 2.0f);
    EXPECT_EQ(searcher.grid_size(), Int3(4, 3, 2));
    EXPECT_EQ(searcher.cell_count(), 4 * 3 * 2);
    EXPECT_EQ(searcher.lower_corner(), Float3(-1.0f, 2.0f, 3.0f));
}

TEST(SpatialHashingSearcher, ViewAliasesTheOwnedBuffers) {
    const SpatialHashingSearcher searcher = make_searcher(Float3(0.0f, 0.0f, 0.0f), 1.0f, Int3(2, 2, 2));

    const SpatialHashingSearcherView view = searcher.view();

    EXPECT_EQ(view.cell_key, searcher.cell_key());
    EXPECT_EQ(view.sorted_index, searcher.indices());
    EXPECT_EQ(view.cell_start, searcher.cell_start());
    EXPECT_EQ(view.cell_end, searcher.cell_end());
}

TEST(SpatialHashingSearcher, UnclassifiedCellRangesAreEmptySentinel) {
    const SpatialHashingSearcher searcher = make_searcher(Float3(0.0f, 0.0f, 0.0f), 1.0f, Int3(3, 2, 1));
    const auto cells                      = static_cast<std::size_t>(searcher.cell_count());

    // No particle has been classified yet: every cell keeps the -1 empty sentinel.
    const std::vector<int> start = to_host(searcher.cell_start(), cells);
    const std::vector<int> end   = to_host(searcher.cell_end(), cells);

    for (std::size_t i = 0; i < cells; ++i) {
        EXPECT_EQ(start[i], -1);
        EXPECT_EQ(end[i], -1);
    }
}

TEST(SpatialHashingSearcherBuilder, DefaultBuilderYieldsUnitGrid) {
    const SpatialHashingSearcher searcher = SpatialHashingSearcher::builder().build();

    EXPECT_FLOAT_EQ(searcher.cell_size(), 1.0f);
    EXPECT_EQ(searcher.grid_size(), Int3(1, 1, 1));
    EXPECT_EQ(searcher.cell_count(), 1);
}

TEST(SpatialHashingSearcherBuilder, RejectsNonPositiveCellSize) {
    EXPECT_THROW(static_cast<void>(SpatialHashingSearcher::builder().with_cell_size(0.0f).build()),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(SpatialHashingSearcher::builder().with_cell_size(-1.0f).build()),
                 std::invalid_argument);
}

TEST(SpatialHashingSearcherBuilder, RejectsGridAxisBelowOne) {
    EXPECT_THROW(static_cast<void>(SpatialHashingSearcher::builder().with_grid_size(Int3(0, 1, 1)).build()),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(SpatialHashingSearcher::builder().with_grid_size(Int3(2, 2, -3)).build()),
                 std::invalid_argument);
}

TEST(SpatialHashingSearcherBuilder, MakeHostSharedBuildsSearcher) {
    const auto searcher = SpatialHashingSearcher::builder()
                              .with_cell_size(2.0f)
                              .with_grid_size(Int3(2, 2, 2))
                              .make_host_shared();

    ASSERT_TRUE(static_cast<bool>(searcher));
    EXPECT_FLOAT_EQ(searcher->cell_size(), 2.0f);
    EXPECT_EQ(searcher->cell_count(), 8);
}

TEST(SpatialHashingSearcher, LinearKeyIsRowMajor) {
    const Int3 grid(4, 3, 2);

    // Row-major: x fastest, then y, then z.
    EXPECT_EQ(SpatialHashingSearcher::linear_key(0, 0, 0, grid), std::uint32_t { 0 });
    EXPECT_EQ(SpatialHashingSearcher::linear_key(1, 0, 0, grid), std::uint32_t { 1 });
    EXPECT_EQ(SpatialHashingSearcher::linear_key(0, 1, 0, grid), std::uint32_t { 4 });
    EXPECT_EQ(SpatialHashingSearcher::linear_key(0, 0, 1, grid), std::uint32_t { 12 });
    EXPECT_EQ(SpatialHashingSearcher::linear_key(Int3(3, 2, 1), grid), std::uint32_t { 3 + 2 * 4 + 1 * 12 });
}

TEST(SpatialHashingSearcher, CellForSameCellHashesEqualAdjacentDiffer) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const float inv = 1.0f;
    const Int3 grid(4, 4, 4);

    const Int3 a = SpatialHashingSearcher::cell_for(Float3(0.2f, 0.2f, 0.2f), lower, inv, grid);
    const Int3 b = SpatialHashingSearcher::cell_for(Float3(0.9f, 0.9f, 0.9f), lower, inv, grid);
    const Int3 c = SpatialHashingSearcher::cell_for(Float3(1.5f, 0.2f, 0.2f), lower, inv, grid);

    // Two points in the same cell hash to the same key; a neighbouring cell differs.
    EXPECT_EQ(a, b);
    EXPECT_EQ(SpatialHashingSearcher::linear_key(a, grid), SpatialHashingSearcher::linear_key(b, grid));
    EXPECT_NE(SpatialHashingSearcher::linear_key(a, grid), SpatialHashingSearcher::linear_key(c, grid));
}

TEST(SpatialHashingSearcher, CellForBoundaryLandsInUpperCell) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const float inv = 1.0f;
    const Int3 grid(4, 4, 4);

    // floor((1.0 - 0)/1) == 1: a point exactly on an interior boundary belongs to the
    // upper of the two cells it separates.
    const Int3 cell = SpatialHashingSearcher::cell_for(Float3(1.0f, 2.0f, 0.0f), lower, inv, grid);
    EXPECT_EQ(cell, Int3(1, 2, 0));
}

TEST(SpatialHashingSearcher, CellForClampsOutOfDomainPoints) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const float inv = 1.0f;
    const Int3 grid(4, 4, 4);

    // Beyond the far corner clamps to the last cell; below the origin clamps to the first.
    EXPECT_EQ(SpatialHashingSearcher::cell_for(Float3(100.0f, 100.0f, 100.0f), lower, inv, grid), Int3(3, 3, 3));
    EXPECT_EQ(SpatialHashingSearcher::cell_for(Float3(-5.0f, -5.0f, -5.0f), lower, inv, grid), Int3(0, 0, 0));
}

TEST(SpatialHashingSearcher, ContainsCellReportsMembershipWithoutClamping) {
    const Int3 grid(4, 4, 4);

    EXPECT_TRUE(SpatialHashingSearcher::contains_cell(Int3(0, 0, 0), grid));
    EXPECT_TRUE(SpatialHashingSearcher::contains_cell(Int3(3, 3, 3), grid));
    EXPECT_FALSE(SpatialHashingSearcher::contains_cell(Int3(-1, 0, 0), grid));
    EXPECT_FALSE(SpatialHashingSearcher::contains_cell(Int3(4, 0, 0), grid));
}

TEST(SpatialHashingSearcher, ClassifyMatchesHandComputedHistogram) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const float cell_size = 1.0f;
    const Int3 grid(2, 2, 2);

    const std::vector<Float3> points {
        Float3(0.5f, 0.5f, 0.5f), // cell (0,0,0) key 0
        Float3(0.1f, 0.1f, 0.1f), // cell (0,0,0) key 0
        Float3(1.5f, 0.5f, 0.5f), // cell (1,0,0) key 1
        Float3(0.5f, 1.5f, 0.5f), // cell (0,1,0) key 2
        Float3(0.5f, 0.5f, 1.5f), // cell (0,0,1) key 4
        Float3(1.5f, 1.5f, 1.5f), // cell (1,1,1) key 7
    };
    const auto count = static_cast<int>(points.size());

    FluidPositionState positions      = make_positions(points);
    SpatialHashingSearcher searcher   = make_searcher(lower, cell_size, grid);
    searcher.classify(&positions, nullptr, count);

    const auto cells             = static_cast<std::size_t>(searcher.cell_count());
    const std::vector<int> hist  = reference_histogram(points, lower, searcher.inverse_cell_size(), grid);
    const std::vector<int> start = to_host(searcher.cell_start(), cells);
    const std::vector<int> end   = to_host(searcher.cell_end(), cells);

    int total = 0;
    for (std::size_t cell = 0; cell < cells; ++cell) {
        if (hist[cell] == 0) {
            // Empty cells keep the sentinel, distinguishing them from a real [n, n) range.
            EXPECT_EQ(start[cell], -1);
            EXPECT_EQ(end[cell], -1);
        } else {
            EXPECT_GE(start[cell], 0);
            EXPECT_EQ(end[cell] - start[cell], hist[cell]);
            total += end[cell] - start[cell];
        }
    }
    EXPECT_EQ(total, count);
}

TEST(SpatialHashingSearcher, ClassifySortedIndicesArePermutationOfInputs) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const Int3 grid(3, 3, 3);

    const std::vector<Float3> points {
        Float3(0.5f, 0.5f, 0.5f),
        Float3(2.5f, 2.5f, 2.5f),
        Float3(1.5f, 0.5f, 2.5f),
        Float3(0.5f, 2.5f, 1.5f),
        Float3(2.5f, 0.5f, 0.5f),
    };
    const auto count = static_cast<int>(points.size());

    FluidPositionState positions    = make_positions(points);
    SpatialHashingSearcher searcher = make_searcher(lower, 1.0f, grid);
    searcher.classify(&positions, nullptr, count);

    std::vector<int> indices = to_host(searcher.indices(), static_cast<std::size_t>(count));
    std::sort(indices.begin(), indices.end());

    std::vector<int> expected(static_cast<std::size_t>(count));
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(indices, expected);
}

TEST(SpatialHashingSearcher, ClassifyGroupsEachCellRunConsistently) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const Int3 grid(2, 2, 2);

    const std::vector<Float3> points {
        Float3(0.5f, 0.5f, 0.5f), // key 0
        Float3(1.5f, 1.5f, 1.5f), // key 7
        Float3(0.2f, 0.2f, 0.2f), // key 0
        Float3(1.5f, 0.5f, 0.5f), // key 1
        Float3(1.7f, 1.7f, 1.7f), // key 7
    };
    const auto count = static_cast<int>(points.size());

    FluidPositionState positions    = make_positions(points);
    SpatialHashingSearcher searcher = make_searcher(lower, 1.0f, grid);
    searcher.classify(&positions, nullptr, count);

    const auto cells                   = static_cast<std::size_t>(searcher.cell_count());
    const std::vector<int> start       = to_host(searcher.cell_start(), cells);
    const std::vector<int> end         = to_host(searcher.cell_end(), cells);
    const std::vector<int> sorted      = to_host(searcher.indices(), static_cast<std::size_t>(count));

    // Every particle in a cell's [start, end) run must actually belong to that cell.
    for (std::size_t cell = 0; cell < cells; ++cell) {
        if (start[cell] < 0) continue;
        for (int slot = start[cell]; slot < end[cell]; ++slot) {
            const Float3 p          = points[static_cast<std::size_t>(sorted[slot])];
            const std::uint32_t key = SpatialHashingSearcher::linear_key(
                SpatialHashingSearcher::cell_for(p, lower, searcher.inverse_cell_size(), grid), grid);
            EXPECT_EQ(key, static_cast<std::uint32_t>(cell));
        }
    }
}

TEST(SpatialHashingSearcher, ClassifyPublishesPerCellCounts) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const Int3 grid(2, 2, 2);

    const std::vector<Float3> points {
        Float3(0.5f, 0.5f, 0.5f), // key 0
        Float3(0.2f, 0.2f, 0.2f), // key 0
        Float3(1.5f, 0.5f, 0.5f), // key 1
        Float3(1.5f, 1.5f, 1.5f), // key 7
    };
    const auto count = static_cast<int>(points.size());

    FluidPositionState positions    = make_positions(points);
    SpatialHashingSearcher searcher = make_searcher(lower, 1.0f, grid);

    const auto cells = static_cast<std::size_t>(searcher.cell_count());
    UniverseNumberParticleState number_particle(cells);
    searcher.classify(&positions, &number_particle, count);

    const std::vector<int> hist = reference_histogram(points, lower, searcher.inverse_cell_size(), grid);
    for (std::size_t cell = 0; cell < cells; ++cell) {
        const float published = number_particle.data()[cell];
        EXPECT_FLOAT_EQ(published, static_cast<float>(hist[cell]));
    }
}

TEST(SpatialHashingSearcher, ClassifyRejectsEmptyInputAndResets) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const Int3 grid(2, 2, 2);

    const std::vector<Float3> points { Float3(0.5f, 0.5f, 0.5f), Float3(1.5f, 1.5f, 1.5f) };
    FluidPositionState populated    = make_positions(points);
    SpatialHashingSearcher searcher = make_searcher(lower, 1.0f, grid);
    searcher.classify(&populated, nullptr, static_cast<int>(points.size()));

    // A non-positive count is unusable input: the searcher falls back to the empty state.
    searcher.classify(&populated, nullptr, 0);

    const auto cells             = static_cast<std::size_t>(searcher.cell_count());
    const std::vector<int> start = to_host(searcher.cell_start(), cells);
    const std::vector<int> end   = to_host(searcher.cell_end(), cells);
    for (std::size_t cell = 0; cell < cells; ++cell) {
        EXPECT_EQ(start[cell], -1);
        EXPECT_EQ(end[cell], -1);
    }
}

TEST(SpatialHashingSearcherView, NeighborQueryMatchesBruteForce) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const Int3 grid(4, 4, 4);

    const std::vector<Float3> points {
        Float3(0.5f, 0.5f, 0.5f),
        Float3(0.7f, 0.4f, 0.6f),
        Float3(1.5f, 0.5f, 0.5f),
        Float3(2.5f, 2.5f, 2.5f),
        Float3(3.5f, 3.5f, 3.5f),
        Float3(0.9f, 0.9f, 0.9f),
    };
    const auto count = static_cast<int>(points.size());

    FluidPositionState positions    = make_positions(points);
    SpatialHashingSearcher searcher = make_searcher(lower, 1.0f, grid);
    searcher.classify(&positions, nullptr, count);

    const Float3 query(0.5f, 0.5f, 0.5f);
    const float radius = 0.6f;

    const std::vector<int> found    = query_neighbor_flags(searcher, positions, count, query, radius);
    const std::vector<int> expected = brute_force_flags(points, query, radius);
    EXPECT_EQ(found, expected);
}

TEST(SpatialHashingSearcherView, NeighborQueryWithNoNeighborsReturnsEmpty) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const Int3 grid(4, 4, 4);

    const std::vector<Float3> points { Float3(0.5f, 0.5f, 0.5f), Float3(1.5f, 1.5f, 1.5f) };
    const auto count = static_cast<int>(points.size());

    FluidPositionState positions    = make_positions(points);
    SpatialHashingSearcher searcher = make_searcher(lower, 1.0f, grid);
    searcher.classify(&positions, nullptr, count);

    // A tiny radius around an empty region finds nothing.
    const std::vector<int> found = query_neighbor_flags(searcher, positions, count, Float3(3.5f, 3.5f, 3.5f), 0.1f);
    EXPECT_EQ(std::count(found.begin(), found.end(), 1), 0);
    EXPECT_EQ(found, brute_force_flags(points, Float3(3.5f, 3.5f, 3.5f), 0.1f));
}

TEST(SpatialHashingSearcherView, NeighborQueryAtDomainCornerStaysInBounds) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const Int3 grid(4, 4, 4);

    const std::vector<Float3> points {
        Float3(3.9f, 3.9f, 3.9f),
        Float3(3.5f, 3.5f, 3.5f),
        Float3(0.5f, 0.5f, 0.5f),
    };
    const auto count = static_cast<int>(points.size());

    FluidPositionState positions    = make_positions(points);
    SpatialHashingSearcher searcher = make_searcher(lower, 1.0f, grid);
    searcher.classify(&positions, nullptr, count);

    // The query sphere reaches past the far corner; cell_for clamping must keep the walk
    // inside the grid rather than indexing out of range.
    const Float3 query(4.0f, 4.0f, 4.0f);
    const float radius = 1.0f;

    const std::vector<int> found    = query_neighbor_flags(searcher, positions, count, query, radius);
    const std::vector<int> expected = brute_force_flags(points, query, radius);
    EXPECT_EQ(found, expected);
}

TEST(SpatialHashingSearcherView, NeighborQuerySpansMultipleCells) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const Int3 grid(4, 4, 4);

    // One point per cell along a line; a radius wider than a cell must gather several.
    const std::vector<Float3> points {
        Float3(0.5f, 1.5f, 1.5f),
        Float3(1.5f, 1.5f, 1.5f),
        Float3(2.5f, 1.5f, 1.5f),
        Float3(3.5f, 1.5f, 1.5f),
        Float3(1.5f, 3.5f, 3.5f),
    };
    const auto count = static_cast<int>(points.size());

    FluidPositionState positions    = make_positions(points);
    SpatialHashingSearcher searcher = make_searcher(lower, 1.0f, grid);
    searcher.classify(&positions, nullptr, count);

    const Float3 query(1.5f, 1.5f, 1.5f);
    const float radius = 1.5f; // spans the three cells centred on the query row

    const std::vector<int> found    = query_neighbor_flags(searcher, positions, count, query, radius);
    const std::vector<int> expected = brute_force_flags(points, query, radius);

    EXPECT_EQ(found, expected);
    // Sanity: the wide radius genuinely picked up more than the query's own cell.
    EXPECT_GT(std::count(found.begin(), found.end(), 1), 1);
}

TEST(SpatialHashingSearcher, RebuildAfterPointSetChangeHasNoStaleState) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const Int3 grid(2, 2, 2);

    SpatialHashingSearcher searcher = make_searcher(lower, 1.0f, grid);
    const auto cells                = static_cast<std::size_t>(searcher.cell_count());

    const std::vector<Float3> first { Float3(0.5f, 0.5f, 0.5f), Float3(0.2f, 0.2f, 0.2f) }; // both key 0
    FluidPositionState first_positions = make_positions(first);
    searcher.classify(&first_positions, nullptr, static_cast<int>(first.size()));

    const std::vector<Float3> second { Float3(1.5f, 1.5f, 1.5f) }; // key 7 only
    FluidPositionState second_positions = make_positions(second);
    searcher.classify(&second_positions, nullptr, static_cast<int>(second.size()));

    const std::vector<int> hist  = reference_histogram(second, lower, searcher.inverse_cell_size(), grid);
    const std::vector<int> start = to_host(searcher.cell_start(), cells);
    const std::vector<int> end   = to_host(searcher.cell_end(), cells);

    for (std::size_t cell = 0; cell < cells; ++cell) {
        if (hist[cell] == 0) {
            // The formerly populated cell 0 must have reverted to the empty sentinel.
            EXPECT_EQ(start[cell], -1);
            EXPECT_EQ(end[cell], -1);
        } else {
            EXPECT_EQ(end[cell] - start[cell], hist[cell]);
        }
    }
}

TEST(SpatialHashingSearcher, MoveConstructionPreservesGeometryAndClassification) {
    const Float3 lower(0.0f, 0.0f, 0.0f);
    const Int3 grid(2, 2, 2);

    const std::vector<Float3> points {
        Float3(0.5f, 0.5f, 0.5f), // key 0
        Float3(0.2f, 0.2f, 0.2f), // key 0
        Float3(1.5f, 1.5f, 1.5f), // key 7
    };
    const auto count = static_cast<int>(points.size());

    FluidPositionState positions   = make_positions(points);
    SpatialHashingSearcher source  = make_searcher(lower, 1.0f, grid);
    source.classify(&positions, nullptr, count);

    const SpatialHashingSearcher moved(std::move(source));

    EXPECT_EQ(moved.grid_size(), grid);
    EXPECT_EQ(moved.cell_count(), 8);
    EXPECT_FLOAT_EQ(moved.cell_size(), 1.0f);

    const auto cells             = static_cast<std::size_t>(moved.cell_count());
    const std::vector<int> hist  = reference_histogram(points, lower, moved.inverse_cell_size(), grid);
    const std::vector<int> start = to_host(moved.cell_start(), cells);
    const std::vector<int> end   = to_host(moved.cell_end(), cells);

    for (std::size_t cell = 0; cell < cells; ++cell) {
        if (hist[cell] == 0) {
            EXPECT_EQ(start[cell], -1);
        } else {
            EXPECT_EQ(end[cell] - start[cell], hist[cell]);
        }
    }
}

TEST(SpatialHashingSearcher, ResetClearsClassificationButKeepsGeometry) {
    const Float3 lower(1.0f, 1.0f, 1.0f);
    const Int3 grid(2, 2, 2);

    const std::vector<Float3> points { Float3(1.5f, 1.5f, 1.5f), Float3(2.5f, 2.5f, 2.5f) };
    FluidPositionState positions    = make_positions(points);
    SpatialHashingSearcher searcher = make_searcher(lower, 1.0f, grid);
    searcher.classify(&positions, nullptr, static_cast<int>(points.size()));

    searcher.reset();

    // Geometry is untouched by reset.
    EXPECT_EQ(searcher.grid_size(), grid);
    EXPECT_EQ(searcher.lower_corner(), lower);

    const auto cells             = static_cast<std::size_t>(searcher.cell_count());
    const std::vector<int> start = to_host(searcher.cell_start(), cells);
    const std::vector<int> end   = to_host(searcher.cell_end(), cells);
    for (std::size_t cell = 0; cell < cells; ++cell) {
        EXPECT_EQ(start[cell], -1);
        EXPECT_EQ(end[cell], -1);
    }
}
