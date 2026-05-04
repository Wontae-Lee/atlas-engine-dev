#include "../utilities/tests_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/geometry/box.h>
#include <atlas/observer/observer.h>
#include <atlas/sink/sink.h>
#include <atlas/source/source.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <testkit/testkit.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

atlas::FluidHostPtr<T>
make_observed_fluid(const atlas::ObserverHostPtr& observer,
                    const std::size_t buffer_size = 64) {
    atlas::HostBuffer<atlas::MaterialProperties<T>> properties(1);

    properties[0] = atlas::MaterialProperties<T>::builder()
                        .with_type(atlas::MaterialType::Molecule)
                        .with_mass(4.651734e-26f)
                        .with_molecular_mass(4.651734e-26f)
                        .with_species_id(0)
                        .with_collision_diameter(4.17e-10f)
                        .build();

    auto fluid = atlas::fluid::Fluid<T>::builder()
                     .with_buffer_size(buffer_size)
                     .with_observer(observer)
                     .make_host_shared();

    fluid->particle_properties() = atlas::DeviceBuffer<atlas::MaterialProperties<T>>(
        properties.begin(),
        properties.end());
    fluid->generators() = atlas::DeviceBuffer<atlas::fluid::GenerateOperator<T>>(
        1,
        atlas::fluid::GenerateOperator<T>(atlas::fluid::GenerateType::uniform, 7u));

    return fluid;
}

atlas::Unit<T>
make_unit(const Vec3& lower = Vec3(-1, -1, -1),
          const Vec3& upper = Vec3(1, 1, 1)) {
    static std::vector<atlas::GeometryHostPtr<T>> geometries;
    static std::vector<atlas::SyncHostPtr<T>> syncs;

    const auto geometry = atlas::geometry::Box<T>::builder()
                              .with_lower_corner(lower)
                              .with_upper_corner(upper)
                              .make_host_shared();

    const auto sync = atlas::physics::Sync<T>::builder().make_host_shared();

    geometries.push_back(geometry);
    syncs.push_back(sync);

    return atlas::physics::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .build();
}

std::string
read_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    return std::string(
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>());
}

} // namespace

TEST(Observer, ExportCsvWritesMetricFiles) {
    const auto observer = atlas::Observer::builder()
                              .with_source_sensor_matrics(2)
                              .with_sink_sensor_matrics(2)
                              .make_host_shared();

    ASSERT_NE(observer, nullptr);

    observer->sensor_matrics<atlas::SourceSensorMatrics>()->record(0, 1, 4);
    observer->sensor_matrics<atlas::SinkSensorMatrics>()->record(2, 3, 7);

    const auto output_dir = std::filesystem::temp_directory_path() / "atlas_observer_export_test";
    std::filesystem::remove_all(output_dir);

    observer->export_csv(output_dir);

    const auto source_path = output_dir / "source_sensor_matrics.csv";
    const auto sink_path   = output_dir / "sink_sensor_matrics.csv";

    ASSERT_TRUE(std::filesystem::exists(source_path));
    ASSERT_TRUE(std::filesystem::exists(sink_path));

    EXPECT_NE(read_file(source_path).find("0,1,4"), std::string::npos);
    EXPECT_NE(read_file(sink_path).find("2,3,7"), std::string::npos);

    std::filesystem::remove_all(output_dir);
}

TEST(Observer, SourceRecordsPerUnitEmissionCounts) {
    const auto observer = atlas::Observer::builder()
                              .with_source_sensor_matrics()
                              .make_host_shared();
    const auto fluid = make_observed_fluid(observer, 128);

    auto source = atlas::fluid::Source<T>::builder()
                      .with_units(atlas::HostBuffer<atlas::Unit<T>> {
                          make_unit(Vec3(-1, -1, -1), Vec3(1, 1, 1)),
                          make_unit(Vec3(-0.5f, -0.5f, -0.5f), Vec3(0.5f, 0.5f, 0.5f)),
                      })
                      .with_fluid(fluid)
                      .with_observer(observer)
                      .with_spawn_types(atlas::HostBuffer<atlas::fluid::SpawnType> {
                          atlas::fluid::SpawnType::Volume,
                          atlas::fluid::SpawnType::Volume,
                      })
                      .with_spawn_operator(atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Volume))
                      .with_spacing(1.0f)
                      .build();

    source.emit();

    const auto* metrics = observer->sensor_matrics<atlas::SourceSensorMatrics>();
    ASSERT_NE(metrics, nullptr);
    const atlas::HostBuffer<atlas::SensorMatrics::Record> records(
        metrics->records().begin(),
        metrics->records().end());
    ASSERT_EQ(records.size(), 2u);

    std::size_t total_emitted = 0;
    for (const auto& record : records) {
        EXPECT_EQ(record.step_index, 0u);
        total_emitted += record.particle_count;
    }

    EXPECT_GT(total_emitted, 0u);
    EXPECT_EQ(total_emitted, fluid->particle_count());
}

TEST(Observer, SinkRecordsPerUnitRemovalCounts) {
    const auto observer = atlas::Observer::builder()
                              .with_sink_sensor_matrics()
                              .make_host_shared();
    const auto fluid = make_observed_fluid(observer, 8);

    auto& positions = fluid->state<atlas::fluid::FluidPositionState<T>>()->data();
    auto& active    = fluid->state<atlas::fluid::FluidActiveState<T>>()->data();

    positions[0] = Vec3(1, 0, 0);
    positions[1] = Vec3(-1, 0, 0);
    active[0]    = 1;
    active[1]    = 1;
    fluid->set_particle_count(2);

    auto sink = atlas::fluid::Sink<T>::builder()
                    .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
                    .with_fluid(fluid)
                    .with_observer(observer)
                    .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> {
                        atlas::fluid::DespawnType::Surface,
                    })
                    .with_despawn_operator(atlas::fluid::DespawnOperator<T>(atlas::fluid::DespawnType::Surface))
                    .build();

    sink.sink();

    const auto* metrics = observer->sensor_matrics<atlas::SinkSensorMatrics>();
    ASSERT_NE(metrics, nullptr);
    const atlas::HostBuffer<atlas::SensorMatrics::Record> records(
        metrics->records().begin(),
        metrics->records().end());
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].step_index, 0u);
    EXPECT_EQ(records[0].unit_index, 0u);
    EXPECT_EQ(records[0].particle_count, 2u);
    EXPECT_EQ(fluid->particle_count(), 0u);
}

TEST(Observer, SinkRecordsFlippedSingleUnitRemovalCounts) {
    const auto observer = atlas::Observer::builder()
                              .with_sink_sensor_matrics()
                              .make_host_shared();
    const auto fluid = make_observed_fluid(observer, 8);

    auto& positions = fluid->state<atlas::fluid::FluidPositionState<T>>()->data();
    auto& active    = fluid->state<atlas::fluid::FluidActiveState<T>>()->data();

    positions[0] = Vec3(2, 0, 0);
    positions[1] = Vec3(-2, 0, 0);
    active[0]    = 1;
    active[1]    = 1;
    fluid->set_particle_count(2);

    auto sink = atlas::fluid::Sink<T>::builder()
                    .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
                    .with_fluid(fluid)
                    .with_observer(observer)
                    .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> {
                        atlas::fluid::DespawnType::Volume,
                    })
                    .with_despawn_operator(atlas::fluid::DespawnOperator<T>(atlas::fluid::DespawnType::Volume))
                    .with_flip(true)
                    .build();

    sink.sink();

    const auto* metrics = observer->sensor_matrics<atlas::SinkSensorMatrics>();
    ASSERT_NE(metrics, nullptr);
    const atlas::HostBuffer<atlas::SensorMatrics::Record> records(
        metrics->records().begin(),
        metrics->records().end());
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].step_index, 0u);
    EXPECT_EQ(records[0].unit_index, 0u);
    EXPECT_EQ(records[0].particle_count, 2u);
    EXPECT_EQ(fluid->particle_count(), 0u);
}
