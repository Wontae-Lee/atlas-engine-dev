#include <atlas/measure/volume_measurer.h>

#include <atlas/buffer/host_buffer.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/box.h>
#include <atlas/math/math.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace {

using atlas::Box;
using atlas::HostBuffer;
using atlas::Quaternion;
using atlas::Sync;
using atlas::Unit;
using atlas::Universe;
using atlas::UniverseVolumeState;
using atlas::Float3;
using atlas::VolumeMeasurer;

auto
make_universe(const HostBuffer<Unit>& measurer_units) {
    return Universe::builder()
        .with_lower_corner(Float3(0.0f, 0.0f, 0.0f))
        .with_upper_corner(Float3(1.0f, 1.0f, 1.0f))
        .with_cell_size(0.5f)
        .with_measurer_units(measurer_units)
        .make_host_shared();
}

Unit
make_box_unit(const Float3& lower,
              const Float3& upper,
              const Float3& translation,
              const Quaternion& orientation = Quaternion(),
              const Float3& velocity = Float3(0.0f, 0.0f, 0.0f)) {
    const auto geometry = Box::builder()
                              .with_lower_corner(lower)
                              .with_upper_corner(upper)
                              .make_host_shared();

    const auto sync = Sync::builder()
                          .with_rigid_pose(translation, orientation)
                          .make_host_shared();

    auto builder = Unit::builder()
                       .with_geometry(atlas::Geometry(*geometry))
                       .with_sync(sync);

    if (velocity.length_squared() > 0.0f) {
        builder.with_velocity(velocity);
    }

    return builder.build();
}

std::uint32_t
linear_key(const int ix, const int iy, const int iz) {
    return static_cast<std::uint32_t>(ix + iy * 3 + iz * 3 * 3);
}

}

TEST(VolumeMeasurer, MeasuresRemainingCellVolume) {
    const auto unit = make_box_unit(
        Float3(-0.2f, -0.2f, -0.2f),
        Float3(0.2f, 0.2f, 0.2f),
        Float3(0.25f, 0.25f, 0.25f));
    const auto universe = make_universe(HostBuffer<Unit> { unit });

    auto measurer = VolumeMeasurer::builder()
                        .with_universe(universe)
                        .with_samples_per_axis(1)
                        .build();

    measurer.measure();

    const auto* state = universe->state<UniverseVolumeState>();
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(state->size(), static_cast<std::size_t>(universe->cell_count()));
    const float occupied_volume = state->data()[linear_key(0, 0, 0)];
    const float free_volume     = state->data()[linear_key(1, 0, 0)];
    EXPECT_FLOAT_EQ(occupied_volume, 0.0f);
    EXPECT_FLOAT_EQ(free_volume, universe->cell_volume());
}

TEST(VolumeMeasurer, StoresUnitsInDeviceBufferAndHandlesEmptyUnits) {
    const auto universe = make_universe(HostBuffer<Unit> {});

    auto measurer = VolumeMeasurer::builder()
                        .with_universe(universe)
                        .with_samples_per_axis(1)
                        .build();

    EXPECT_TRUE(measurer.units().empty());

    measurer.measure();

    const auto* state = universe->state<UniverseVolumeState>();
    ASSERT_NE(state, nullptr);
    const float first_volume = state->data()[linear_key(0, 0, 0)];
    const float last_volume  = state->data()[linear_key(1, 1, 1)];
    EXPECT_FLOAT_EQ(first_volume, universe->cell_volume());
    EXPECT_FLOAT_EQ(last_volume, universe->cell_volume());
}

TEST(VolumeMeasurer, AdvancesUnitsByDtBeforeMeasuring) {
    const auto unit = make_box_unit(
        Float3(-0.2f, -0.2f, -0.2f),
        Float3(0.2f, 0.2f, 0.2f),
        Float3(-0.5f, 0.25f, 0.25f),
        Quaternion(),
        Float3(0.75f, 0.0f, 0.0f));
    const auto universe = make_universe(HostBuffer<Unit> { unit });

    auto measurer = VolumeMeasurer::builder()
                        .with_universe(universe)
                        .with_samples_per_axis(1)
                        .build();

    measurer.measure(1.0f);

    const auto* state = universe->state<UniverseVolumeState>();
    ASSERT_NE(state, nullptr);
    const float occupied_volume = state->data()[linear_key(0, 0, 0)];
    EXPECT_FLOAT_EQ(occupied_volume, 0.0f);
}

TEST(VolumeMeasurer, AppliesUnitRotationDuringInsideTest) {
    const auto unit = make_box_unit(
        Float3(-0.3f, -0.05f, -0.1f),
        Float3(0.3f, 0.05f, 0.1f),
        Float3(0.75f, 0.5f, 0.75f),
        Quaternion(Float3(0.0f, 0.0f, 1.0f), atlas::pi * 0.5f));
    const auto universe = make_universe(HostBuffer<Unit> { unit });

    auto measurer = VolumeMeasurer::builder()
                        .with_universe(universe)
                        .with_samples_per_axis(1)
                        .build();

    measurer.measure();

    const auto* state = universe->state<UniverseVolumeState>();
    ASSERT_NE(state, nullptr);
    const float occupied_volume = state->data()[linear_key(1, 1, 1)];
    EXPECT_FLOAT_EQ(occupied_volume, 0.0f);
}

TEST(VolumeMeasurer, ParallelCellPassMeasuresIndependentCells) {
    const auto first_unit = make_box_unit(
        Float3(-0.2f, -0.2f, -0.2f),
        Float3(0.2f, 0.2f, 0.2f),
        Float3(0.25f, 0.25f, 0.25f));
    const auto second_unit = make_box_unit(
        Float3(-0.2f, -0.2f, -0.2f),
        Float3(0.2f, 0.2f, 0.2f),
        Float3(0.75f, 0.25f, 0.25f));
    const auto universe = make_universe(HostBuffer<Unit> { first_unit, second_unit });

    auto measurer = VolumeMeasurer::builder()
                        .with_universe(universe)
                        .with_samples_per_axis(1)
                        .build();

    measurer.measure();

    const auto* state = universe->state<UniverseVolumeState>();
    ASSERT_NE(state, nullptr);
    const float first_volume  = state->data()[linear_key(0, 0, 0)];
    const float second_volume = state->data()[linear_key(1, 0, 0)];
    const float free_volume   = state->data()[linear_key(0, 1, 0)];
    EXPECT_FLOAT_EQ(first_volume, 0.0f);
    EXPECT_FLOAT_EQ(second_volume, 0.0f);
    EXPECT_FLOAT_EQ(free_volume, universe->cell_volume());
}

TEST(VolumeMeasurer, OverlappingUnitsSubtractSampleVolumeOnlyOnce) {
    const auto first_unit = make_box_unit(
        Float3(-0.2f, -0.2f, -0.2f),
        Float3(0.2f, 0.2f, 0.2f),
        Float3(0.25f, 0.25f, 0.25f));
    const auto second_unit = make_box_unit(
        Float3(-0.2f, -0.2f, -0.2f),
        Float3(0.2f, 0.2f, 0.2f),
        Float3(0.25f, 0.25f, 0.25f));
    const auto universe = make_universe(HostBuffer<Unit> { first_unit, second_unit });

    auto measurer = VolumeMeasurer::builder()
                        .with_universe(universe)
                        .with_samples_per_axis(1)
                        .build();

    measurer.measure();

    const auto* state = universe->state<UniverseVolumeState>();
    ASSERT_NE(state, nullptr);
    const float occupied_volume = state->data()[linear_key(0, 0, 0)];
    EXPECT_FLOAT_EQ(occupied_volume, 0.0f);
}
