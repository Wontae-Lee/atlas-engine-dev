#include "../utilities/test_utils.h"

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

using atlas::Box;
using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::GenerateOperator;
using atlas::GenerateType;
using atlas::GeometryHostPtr;
using atlas::HostBuffer;
using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::Observer;
using atlas::ObserverHostPtr;
using atlas::SensorMatrics;
using atlas::Sink;
using atlas::SinkSensorMatrics;
using atlas::Source;
using atlas::SourceSensorMatrics;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::Unit;
using atlas::Vector3F;
using atlas::fluid::DespawnOperator;
using atlas::fluid::DespawnType;
using atlas::fluid::FluidActiveState;
using atlas::fluid::FluidPositionState;
using atlas::fluid::SpawnOperator;
using atlas::fluid::SpawnType;

FluidHostPtr<float>
make_observed_fluid(const ObserverHostPtr& observer,
                    const std::size_t buffer_size = 64) {
    HostBuffer<MaterialProperties<float>> properties(1);

    properties[0] = MaterialProperties<float>::builder()
                        .with_type(MaterialType::Molecule)
                        .with_mass(4.651734e-26f)
                        .with_molecular_mass(4.651734e-26f)
                        .with_species_id(0)
                        .with_collision_diameter(4.17e-10f)
                        .build();

    auto fluid = Fluid<float>::builder()
                     .with_buffer_size(buffer_size)
                     .with_observer(observer)
                     .make_host_shared();

    fluid->particle_properties() = DeviceBuffer<MaterialProperties<float>>(
        properties.begin(),
        properties.end());
    fluid->generators() = DeviceBuffer<GenerateOperator<float>>(
        1,
        GenerateOperator<float>(GenerateType::uniform, 7u));

    return fluid;
}

Unit<float>
make_unit(const Vector3F& lower = Vector3F(-1, -1, -1),
          const Vector3F& upper = Vector3F(1, 1, 1)) {
    static std::vector<GeometryHostPtr<float>> geometries;
    static std::vector<SyncHostPtr<float>> syncs;

    const auto geometry = Box<float>::builder()
                              .with_lower_corner(lower)
                              .with_upper_corner(upper)
                              .make_host_shared();

    const auto sync = Sync<float>::builder().make_host_shared();

    geometries.push_back(geometry);
    syncs.push_back(sync);

    return Unit<float>::builder()
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
    const auto observer = Observer::builder()
                              .with_source_sensor_matrics(2)
                              .with_sink_sensor_matrics(2)
                              .make_host_shared();

    ASSERT_NE(observer, nullptr);

    observer->sensor_matrics<SourceSensorMatrics>()->record(0, 1, 4);
    observer->sensor_matrics<SinkSensorMatrics>()->record(2, 3, 7);

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
    const auto observer = Observer::builder()
                              .with_source_sensor_matrics()
                              .make_host_shared();
    const auto fluid = make_observed_fluid(observer, 128);

    auto source = Source<float>::builder()
                      .with_units(HostBuffer<Unit<float>> {
                          make_unit(Vector3F(-1, -1, -1), Vector3F(1, 1, 1)),
                          make_unit(Vector3F(-0.5f, -0.5f, -0.5f), Vector3F(0.5f, 0.5f, 0.5f)),
                      })
                      .with_fluid(fluid)
                      .with_observer(observer)
                      .with_spawn_types(HostBuffer<SpawnType> {
                          SpawnType::Volume,
                          SpawnType::Volume,
                      })
                      .with_spawn_operator(SpawnOperator<float>(SpawnType::Volume))
                      .with_spacing(1.0f)
                      .build();

    source.emit();

    const auto* metrics = observer->sensor_matrics<SourceSensorMatrics>();
    ASSERT_NE(metrics, nullptr);
    const HostBuffer<SensorMatrics::Record> records(
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
    const auto observer = Observer::builder()
                              .with_sink_sensor_matrics()
                              .make_host_shared();
    const auto fluid = make_observed_fluid(observer, 8);

    auto& positions = fluid->state<FluidPositionState<float>>()->data();
    auto& active    = fluid->state<FluidActiveState<float>>()->data();

    positions[0] = Vector3F(1, 0, 0);
    positions[1] = Vector3F(-1, 0, 0);
    active[0]    = 1;
    active[1]    = 1;
    fluid->set_particle_count(2);

    auto sink = Sink<float>::builder()
                    .with_units(HostBuffer<Unit<float>> { make_unit() })
                    .with_fluid(fluid)
                    .with_observer(observer)
                    .with_despawn_types(HostBuffer<DespawnType> {
                        DespawnType::Surface,
                    })
                    .with_despawn_operator(DespawnOperator<float>(DespawnType::Surface))
                    .build();

    sink.sink();

    const auto* metrics = observer->sensor_matrics<SinkSensorMatrics>();
    ASSERT_NE(metrics, nullptr);
    const HostBuffer<SensorMatrics::Record> records(
        metrics->records().begin(),
        metrics->records().end());
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].step_index, 0u);
    EXPECT_EQ(records[0].unit_index, 0u);
    EXPECT_EQ(records[0].particle_count, 2u);
    EXPECT_EQ(fluid->particle_count(), 0u);
}

TEST(Observer, SinkRecordsFlippedSingleUnitRemovalCounts) {
    const auto observer = Observer::builder()
                              .with_sink_sensor_matrics()
                              .make_host_shared();
    const auto fluid = make_observed_fluid(observer, 8);

    auto& positions = fluid->state<FluidPositionState<float>>()->data();
    auto& active    = fluid->state<FluidActiveState<float>>()->data();

    positions[0] = Vector3F(2, 0, 0);
    positions[1] = Vector3F(-2, 0, 0);
    active[0]    = 1;
    active[1]    = 1;
    fluid->set_particle_count(2);

    auto sink = Sink<float>::builder()
                    .with_units(HostBuffer<Unit<float>> { make_unit() })
                    .with_fluid(fluid)
                    .with_observer(observer)
                    .with_despawn_types(HostBuffer<DespawnType> {
                        DespawnType::Volume,
                    })
                    .with_despawn_operator(DespawnOperator<float>(DespawnType::Volume))
                    .with_flip(true)
                    .build();

    sink.sink();

    const auto* metrics = observer->sensor_matrics<SinkSensorMatrics>();
    ASSERT_NE(metrics, nullptr);
    const HostBuffer<SensorMatrics::Record> records(
        metrics->records().begin(),
        metrics->records().end());
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].step_index, 0u);
    EXPECT_EQ(records[0].unit_index, 0u);
    EXPECT_EQ(records[0].particle_count, 2u);
    EXPECT_EQ(fluid->particle_count(), 0u);
}
