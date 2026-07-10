#include <atlas/observer/observer.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using atlas::Fluid;
using atlas::HostBuffer;
using atlas::Observer;
using atlas::Universe;

/** @brief Copies a counter buffer back to the host for value inspection. */
std::vector<int>
read_counter(const atlas::DeviceBuffer<int>& counter) {
    const HostBuffer<int> host(counter.begin(), counter.end());
    return std::vector<int>(host.begin(), host.end());
}

/** @brief A minimal, empty fluid so observe() skips the fluid CSV. */
Fluid
make_empty_fluid() {
    return Fluid::builder().with_buffer_size(1).build();
}

/** @brief The default unit-box universe (8 cells, no registered state columns). */
Universe
make_universe() {
    return Universe::builder().build();
}

/** @brief Reads an entire file into a string. */
std::string
slurp(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

}

TEST(Observer, DefaultObserverHasNoIntervalOrCounters) {
    const Observer observer;

    EXPECT_EQ(observer.interval(), 0u);
    EXPECT_EQ(observer.species_count(), 0u);
    EXPECT_TRUE(observer.spawned().empty());
    EXPECT_TRUE(observer.despawned().empty());
    EXPECT_TRUE(observer.output_directory().empty());
}

TEST(Observer, BuilderCarriesIntervalAndOutputDirectory) {
    const std::filesystem::path out = "some/output/root";

    const Observer observer = Observer::builder()
                                  .with_interval(7)
                                  .with_output_directory(out)
                                  .build();

    EXPECT_EQ(observer.interval(), 7u);
    EXPECT_EQ(observer.output_directory(), out);
    // Counters are not sized by the builder.
    EXPECT_TRUE(observer.spawned().empty());
}

TEST(Observer, MakeHostSharedProducesAConfiguredObserver) {
    const auto observer = Observer::builder().with_interval(3).make_host_shared();

    ASSERT_TRUE(static_cast<bool>(observer));
    EXPECT_EQ(observer->interval(), 3u);
}

TEST(Observer, ResizeCountersAllocatesRowMajorAndZeroFills) {
    Observer observer = Observer::builder().build();

    observer.resize_counters(2, 3, 4);

    EXPECT_EQ(observer.species_count(), 4u);
    EXPECT_EQ(observer.spawned().size(), 2u * 4u);
    EXPECT_EQ(observer.despawned().size(), 3u * 4u);

    for (const int value : read_counter(observer.spawned())) {
        EXPECT_EQ(value, 0);
    }
    for (const int value : read_counter(observer.despawned())) {
        EXPECT_EQ(value, 0);
    }
}

TEST(Observer, ResetCountersZeroesInPlaceWithoutResizing) {
    Observer observer = Observer::builder().build();
    observer.resize_counters(2, 1, 3);

    // Write known values, then clear.
    const std::vector<int> values { 1, 2, 3, 4, 5, 6 };
    const HostBuffer<int> spawn_values(values.begin(), values.end());
    observer.spawned() = atlas::DeviceBuffer<int>(spawn_values.begin(), spawn_values.end());

    observer.reset_counters();

    ASSERT_EQ(observer.spawned().size(), 6u);
    ASSERT_EQ(observer.despawned().size(), 3u);
    for (const int value : read_counter(observer.spawned())) {
        EXPECT_EQ(value, 0);
    }
}

TEST(Observer, ObserveIsANoOpWhenTheIntervalIsZero) {
    const std::filesystem::path root
        = std::filesystem::temp_directory_path() / "atlas_observer_zero_interval";
    std::filesystem::remove_all(root);

    const Observer observer = Observer::builder()
                                  .with_interval(0)
                                  .with_output_directory(root)
                                  .build();

    observer.observe(make_empty_fluid(), make_universe(), 10);

    EXPECT_FALSE(std::filesystem::exists(root / "data"));
    std::filesystem::remove_all(root);
}

TEST(Observer, ObserveSkipsStepsThatAreNotAMultipleOfTheInterval) {
    const std::filesystem::path root
        = std::filesystem::temp_directory_path() / "atlas_observer_off_interval";
    std::filesystem::remove_all(root);

    const Observer observer = Observer::builder()
                                  .with_interval(5)
                                  .with_output_directory(root)
                                  .build();

    observer.observe(make_empty_fluid(), make_universe(), 3);

    EXPECT_FALSE(std::filesystem::exists(root / "data"));
    std::filesystem::remove_all(root);
}

TEST(Observer, ObserveSerializesSpawnCountersToCsv) {
    const std::filesystem::path root
        = std::filesystem::temp_directory_path() / "atlas_observer_spawn_csv";
    std::filesystem::remove_all(root);

    Observer observer = Observer::builder()
                            .with_interval(5)
                            .with_output_directory(root)
                            .build();

    // Two sources, three species, no sinks: spawn matrix [source][species].
    observer.resize_counters(2, 0, 3);
    const std::vector<int> values { 1, 2, 3, 4, 5, 6 };
    const HostBuffer<int> spawn_values(values.begin(), values.end());
    observer.spawned() = atlas::DeviceBuffer<int>(spawn_values.begin(), spawn_values.end());

    observer.observe(make_empty_fluid(), make_universe(), 5);

    const std::filesystem::path source_csv = root / "data" / "source_5.csv";
    ASSERT_TRUE(std::filesystem::exists(source_csv));

    const std::string text = slurp(source_csv);

    // Header names one column per species, then one row per source with its tallies.
    EXPECT_NE(text.find("source,species_0,species_1,species_2"), std::string::npos);
    EXPECT_NE(text.find("0,1,2,3"), std::string::npos);
    EXPECT_NE(text.find("1,4,5,6"), std::string::npos);

    // No sinks were configured, so no sink file is produced.
    EXPECT_FALSE(std::filesystem::exists(root / "data" / "sink_5.csv"));

    std::filesystem::remove_all(root);
}
