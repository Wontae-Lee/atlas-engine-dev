#include "../utilities/test_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/math/constants.h>
#include <atlas/measure/volume_measurer.h>
#include <atlas/sync/sync.h>
#include <atlas/universe/universe.h>

#include <testkit/testkit.h>

namespace {

using atlas::Box;
using atlas::GeometryHostPtr;
using atlas::HostBuffer;
using atlas::Quaternion;
using atlas::Sync;
using atlas::Unit;
using atlas::Universe;
using atlas::Vector3F;
using atlas::VolumeMeasurer;
using atlas::UniverseVolumeState;

auto
make_universe() {
    return Universe<float>::builder()
        .with_lower_corner(Vector3F(0, 0, 0))
        .with_upper_corner(Vector3F(1, 1, 1))
        .with_cell_size(0.5f)
        .make_host_shared();
}

Unit<float>
make_box_unit(const Vector3F& lower,
              const Vector3F& upper,
              const Vector3F& translation,
              const Quaternion<float>& orientation = Quaternion<float>(),
              const Vector3F& velocity = Vector3F(0, 0, 0)) {
    static HostBuffer<GeometryHostPtr<float>> geometry_owners;

    const auto geometry = Box<float>::builder()
                              .with_lower_corner(lower)
                              .with_upper_corner(upper)
                              .make_host_shared();
    geometry_owners.push_back(geometry);

    const auto sync = Sync<float>::builder()
                          .with_rigid_pose(translation, orientation)
                          .make_host_shared();

    auto builder = Unit<float>::builder()
                       .with_geometry(geometry)
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

} // namespace

TEST(VolumeMeasurer, MeasuresRemainingCellVolume) {
    const auto universe = make_universe();
    const auto unit = make_box_unit(
        Vector3F(-0.2f, -0.2f, -0.2f),
        Vector3F(0.2f, 0.2f, 0.2f),
        Vector3F(0.25f, 0.25f, 0.25f));

    auto measurer = VolumeMeasurer<float>::builder()
                        .with_universe(universe)
                        .with_units(HostBuffer<Unit<float>> { unit })
                        .with_samples_per_axis(1)
                        .build();

    measurer.measure();

    const auto* state = universe->state<UniverseVolumeState<float>>();
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(state->size(), static_cast<std::size_t>(universe->number_of_cells()));
    EXPECT_FLOAT_EQ(state->data()[linear_key(0, 0, 0)], 0.0f);
    EXPECT_FLOAT_EQ(state->data()[linear_key(1, 0, 0)], universe->cell_volume());
}

TEST(VolumeMeasurer, StoresUnitsInDeviceBufferAndHandlesEmptyUnits) {
    const auto universe = make_universe();

    auto measurer = VolumeMeasurer<float>::builder()
                        .with_universe(universe)
                        .with_units(HostBuffer<Unit<float>> {})
                        .with_samples_per_axis(1)
                        .build();

    EXPECT_TRUE(measurer.units().empty());

    measurer.measure();

    const auto* state = universe->state<UniverseVolumeState<float>>();
    ASSERT_NE(state, nullptr);
    EXPECT_FLOAT_EQ(state->data()[linear_key(0, 0, 0)], universe->cell_volume());
    EXPECT_FLOAT_EQ(state->data()[linear_key(1, 1, 1)], universe->cell_volume());
}

TEST(VolumeMeasurer, AdvancesUnitsByDtBeforeMeasuring) {
    const auto universe = make_universe();
    const auto unit = make_box_unit(
        Vector3F(-0.2f, -0.2f, -0.2f),
        Vector3F(0.2f, 0.2f, 0.2f),
        Vector3F(-0.5f, 0.25f, 0.25f),
        Quaternion<float>(),
        Vector3F(0.75f, 0.0f, 0.0f));

    auto measurer = VolumeMeasurer<float>::builder()
                        .with_universe(universe)
                        .with_units(HostBuffer<Unit<float>> { unit })
                        .with_samples_per_axis(1)
                        .build();

    measurer.measure(1.0f);

    const auto* state = universe->state<UniverseVolumeState<float>>();
    ASSERT_NE(state, nullptr);
    EXPECT_FLOAT_EQ(state->data()[linear_key(0, 0, 0)], 0.0f);
}

TEST(VolumeMeasurer, AppliesUnitRotationDuringInsideTest) {
    const auto universe = make_universe();
    const auto unit = make_box_unit(
        Vector3F(-0.3f, -0.05f, -0.1f),
        Vector3F(0.3f, 0.05f, 0.1f),
        Vector3F(0.75f, 0.5f, 0.75f),
        Quaternion<float>(Vector3F(0, 0, 1), static_cast<float>(atlas::pi) * 0.5f));

    auto measurer = VolumeMeasurer<float>::builder()
                        .with_universe(universe)
                        .with_units(HostBuffer<Unit<float>> { unit })
                        .with_samples_per_axis(1)
                        .build();

    measurer.measure();

    const auto* state = universe->state<UniverseVolumeState<float>>();
    ASSERT_NE(state, nullptr);
    EXPECT_FLOAT_EQ(state->data()[linear_key(1, 1, 1)], 0.0f);
}

TEST(VolumeMeasurer, ParallelCellPassMeasuresIndependentCells) {
    const auto universe = make_universe();
    const auto first_unit = make_box_unit(
        Vector3F(-0.2f, -0.2f, -0.2f),
        Vector3F(0.2f, 0.2f, 0.2f),
        Vector3F(0.25f, 0.25f, 0.25f));
    const auto second_unit = make_box_unit(
        Vector3F(-0.2f, -0.2f, -0.2f),
        Vector3F(0.2f, 0.2f, 0.2f),
        Vector3F(0.75f, 0.25f, 0.25f));

    auto measurer = VolumeMeasurer<float>::builder()
                        .with_universe(universe)
                        .with_units(HostBuffer<Unit<float>> { first_unit, second_unit })
                        .with_samples_per_axis(1)
                        .build();

    measurer.measure();

    const auto* state = universe->state<UniverseVolumeState<float>>();
    ASSERT_NE(state, nullptr);
    EXPECT_FLOAT_EQ(state->data()[linear_key(0, 0, 0)], 0.0f);
    EXPECT_FLOAT_EQ(state->data()[linear_key(1, 0, 0)], 0.0f);
    EXPECT_FLOAT_EQ(state->data()[linear_key(0, 1, 0)], universe->cell_volume());
}

TEST(VolumeMeasurer, OverlappingUnitsSubtractSampleVolumeOnlyOnce) {
    const auto universe = make_universe();
    const auto first_unit = make_box_unit(
        Vector3F(-0.2f, -0.2f, -0.2f),
        Vector3F(0.2f, 0.2f, 0.2f),
        Vector3F(0.25f, 0.25f, 0.25f));
    const auto second_unit = make_box_unit(
        Vector3F(-0.2f, -0.2f, -0.2f),
        Vector3F(0.2f, 0.2f, 0.2f),
        Vector3F(0.25f, 0.25f, 0.25f));

    auto measurer = VolumeMeasurer<float>::builder()
                        .with_universe(universe)
                        .with_units(HostBuffer<Unit<float>> { first_unit, second_unit })
                        .with_samples_per_axis(1)
                        .build();

    measurer.measure();

    const auto* state = universe->state<UniverseVolumeState<float>>();
    ASSERT_NE(state, nullptr);
    EXPECT_FLOAT_EQ(state->data()[linear_key(0, 0, 0)], 0.0f);
}
