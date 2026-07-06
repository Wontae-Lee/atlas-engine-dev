#include <atlas/observer/observer.h>

#include <atlas/fluid/fluid.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/box.h>
#include <atlas/sink/sink.h>
#include <atlas/source/source.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

using atlas::Box;
using atlas::Despawn;
using atlas::DespawnType;
using atlas::Fluid;
using atlas::FluidActiveState;
using atlas::FluidHostPtr;
using atlas::FluidPositionState;
using atlas::HostBuffer;
using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::MaxwellBoltzmannGenerator;
using atlas::Observer;
using atlas::ObserverHostPtr;
using atlas::SensorMetrics;
using atlas::Sink;
using atlas::SinkSensorMetrics;
using atlas::Source;
using atlas::SourceSensorMetrics;
using atlas::Spawn;
using atlas::SpawnType;
using atlas::Sync;
using atlas::Unit;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Float3;

FluidHostPtr
make_observed_fluid(const ObserverHostPtr& observer,
                    const std::size_t buffer_size = 64) {
    HostBuffer<MaterialProperties> properties(1);
    HostBuffer<atlas::GeneratorHostPtr> generators(1);

    properties[0] = MaterialProperties::builder()
                        .with_type(MaterialType::molecule)
                        .with_mass(4.651734e-26f)
                        .with_molecular_mass(4.651734e-26f)
                        .with_species_id(0)
                        .with_reference_diameter(4.17e-10f)
                        .build();

    generators[0] = MaxwellBoltzmannGenerator::builder()
                        .with_temperature(300.0f)
                        .with_molecular_mass(4.651734e-26f)
                        .with_bulk_velocity(Float3(0.0f, 0.0f, 0.0f))
                        .with_seed(7u)
                        .make_host_shared();

    return Fluid::builder()
        .with_buffer_size(buffer_size)
        .with_properties(properties)
        .with_generators(generators)
        .with_observer(observer)
        .make_host_shared();
}

Unit
make_unit(const Float3& lower = Float3(-1.0f, -1.0f, -1.0f),
          const Float3& upper = Float3(1.0f, 1.0f, 1.0f)) {
    const auto geometry = Box::builder()
                              .with_lower_corner(lower)
                              .with_upper_corner(upper)
                              .make_host_shared();

    const auto sync = Sync::builder().make_host_shared();

    return Unit::builder()
        .with_geometry(atlas::Geometry(*geometry))
        .with_sync(sync)
        .build();
}

UniverseHostPtr
make_source_universe(const HostBuffer<Unit>& source_units) {
    return Universe::builder()
        .with_lower_corner(Float3(-10.0f, -10.0f, -10.0f))
        .with_upper_corner(Float3(10.0f, 10.0f, 10.0f))
        .with_cell_size(1.0f)
        .with_source_units(source_units)
        .make_host_shared();
}

UniverseHostPtr
make_sink_universe(const HostBuffer<Unit>& sink_units) {
    return Universe::builder()
        .with_lower_corner(Float3(-10.0f, -10.0f, -10.0f))
        .with_upper_corner(Float3(10.0f, 10.0f, 10.0f))
        .with_cell_size(1.0f)
        .with_sink_units(sink_units)
        .make_host_shared();
}

std::string
read_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    return std::string(
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>());
}

}

TEST(Observer, ExportCsvWritesMetricFiles) {
    const auto observer = Observer::builder()
                              .with_source_sensor_metrics(2)
                              .with_sink_sensor_metrics(2)
                              .make_host_shared();

    ASSERT_NE(observer, nullptr);

    observer->sensor_metrics<SourceSensorMetrics>()->record(0, 1, 4);
    observer->sensor_metrics<SinkSensorMetrics>()->record(2, 3, 7);

    const auto output_dir = std::filesystem::temp_directory_path() / "atlas_observer_export_test";
    std::filesystem::remove_all(output_dir);

    observer->export_csv(output_dir);

    const auto source_path = output_dir / "source_sensor_metrics.csv";
    const auto sink_path   = output_dir / "sink_sensor_metrics.csv";

    ASSERT_TRUE(std::filesystem::exists(source_path));
    ASSERT_TRUE(std::filesystem::exists(sink_path));

    EXPECT_NE(read_file(source_path).find("0,1,4"), std::string::npos);
    EXPECT_NE(read_file(sink_path).find("2,3,7"), std::string::npos);

    std::filesystem::remove_all(output_dir);
}

TEST(Observer, SourceRecordsPerUnitEmissionCounts) {
    const auto observer = Observer::builder()
                              .with_source_sensor_metrics()
                              .make_host_shared();
    const auto fluid = make_observed_fluid(observer, 128);

    auto source = Source::builder()
                      .with_universe(make_source_universe(HostBuffer<Unit> {
                          make_unit(Float3(-1.0f, -1.0f, -1.0f), Float3(1.0f, 1.0f, 1.0f)),
                          make_unit(Float3(-0.5f, -0.5f, -0.5f), Float3(0.5f, 0.5f, 0.5f)),
                      }))
                      .with_fluid(fluid)
                      .with_observer(observer)
                      .with_spawn_types(HostBuffer<SpawnType> {
                          SpawnType::volume,
                          SpawnType::volume,
                      })
                      .with_spawn_operator(Spawn(SpawnType::volume))
                      .with_spacing(1.0f)
                      .build();

    source.emit();

    const auto* metrics = observer->sensor_metrics<SourceSensorMetrics>();
    ASSERT_NE(metrics, nullptr);
    const auto& records = metrics->records();
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
                              .with_sink_sensor_metrics()
                              .make_host_shared();
    const auto fluid = make_observed_fluid(observer, 8);

    auto& positions = fluid->state<FluidPositionState>()->data();
    auto& active    = fluid->state<FluidActiveState>()->data();

    positions[0] = Float3(1.0f, 0.0f, 0.0f);
    positions[1] = Float3(-1.0f, 0.0f, 0.0f);
    active[0]    = 1;
    active[1]    = 1;
    fluid->set_particle_count(2);

    auto sink = Sink::builder()
                    .with_universe(make_sink_universe(HostBuffer<Unit> { make_unit() }))
                    .with_fluid(fluid)
                    .with_observer(observer)
                    .with_despawn_types(HostBuffer<DespawnType> {
                        DespawnType::surface,
                    })
                    .with_despawn_operator(Despawn(DespawnType::surface))
                    .build();

    sink.sink();

    const auto* metrics = observer->sensor_metrics<SinkSensorMetrics>();
    ASSERT_NE(metrics, nullptr);
    const auto& records = metrics->records();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].step_index, 0u);
    EXPECT_EQ(records[0].unit_index, 0u);
    EXPECT_EQ(records[0].particle_count, 2u);
    EXPECT_EQ(fluid->particle_count(), 0u);
}

TEST(Observer, SinkRecordsFlippedSingleUnitRemovalCounts) {
    const auto observer = Observer::builder()
                              .with_sink_sensor_metrics()
                              .make_host_shared();
    const auto fluid = make_observed_fluid(observer, 8);

    auto& positions = fluid->state<FluidPositionState>()->data();
    auto& active    = fluid->state<FluidActiveState>()->data();

    positions[0] = Float3(2.0f, 0.0f, 0.0f);
    positions[1] = Float3(-2.0f, 0.0f, 0.0f);
    active[0]    = 1;
    active[1]    = 1;
    fluid->set_particle_count(2);

    auto sink = Sink::builder()
                    .with_universe(make_sink_universe(HostBuffer<Unit> { make_unit() }))
                    .with_fluid(fluid)
                    .with_observer(observer)
                    .with_despawn_types(HostBuffer<DespawnType> {
                        DespawnType::volume,
                    })
                    .with_despawn_operator(Despawn(DespawnType::volume))
                    .with_flip(true)
                    .build();

    sink.sink();

    const auto* metrics = observer->sensor_metrics<SinkSensorMetrics>();
    ASSERT_NE(metrics, nullptr);
    const auto& records = metrics->records();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].step_index, 0u);
    EXPECT_EQ(records[0].unit_index, 0u);
    EXPECT_EQ(records[0].particle_count, 2u);
    EXPECT_EQ(fluid->particle_count(), 0u);
}
