#include <atlas/serialization/protobuf_snapshot.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

using atlas::FluidBinarySnapshot;
using atlas::UniverseBinarySnapshot;
using atlas::load_fluid_binary;
using atlas::load_universe_binary;

// A path that is guaranteed not to resolve to a readable file, so the loaders
// hit their "failed to open snapshot file" branch.
constexpr const char* kMissingPath = "/nonexistent-atlas-dir/definitely-missing-snapshot.bin";

// Writes a few non-protobuf bytes to a scratch file and removes it on scope
// exit, so a loader can be driven down its parse/validation failure path.
class CorruptSnapshotFile {
public:
    explicit CorruptSnapshotFile(std::string path)
        : _path(std::move(path)) {
        std::ofstream out(_path, std::ios::binary | std::ios::trunc);
        out << "not a valid protobuf snapshot payload";
    }

    ~CorruptSnapshotFile() {
        std::remove(_path.c_str());
    }

    const std::string&
    path() const {
        return _path;
    }

private:
    std::string _path;
};

}

TEST(FluidBinarySnapshot, DefaultConstructsEmpty) {
    const FluidBinarySnapshot snapshot;

    EXPECT_EQ(snapshot.buffer_size, 0u);
    EXPECT_EQ(snapshot.particle_count, 0u);
    EXPECT_FLOAT_EQ(snapshot.statistical_weight, 1.0f);
    EXPECT_FALSE(snapshot.positions.has_value());
    EXPECT_FALSE(snapshot.velocities.has_value());
    EXPECT_FALSE(snapshot.species.has_value());
    EXPECT_FALSE(snapshot.active.has_value());
    EXPECT_FALSE(snapshot.translational_energy.has_value());
}

TEST(UniverseBinarySnapshot, DefaultConstructsEmpty) {
    const UniverseBinarySnapshot snapshot;

    EXPECT_FLOAT_EQ(snapshot.cell_size, 1.0f);
    EXPECT_FALSE(snapshot.temperature.has_value());
    EXPECT_FALSE(snapshot.bulk_velocity.has_value());
    EXPECT_FALSE(snapshot.collision_count.has_value());
    EXPECT_FALSE(snapshot.allocated_solver.has_value());
}

TEST(LoadFluidBinary, ThrowsOnMissingFile) {
    EXPECT_THROW(load_fluid_binary(kMissingPath), std::runtime_error);
}

TEST(LoadUniverseBinary, ThrowsOnMissingFile) {
    EXPECT_THROW(load_universe_binary(kMissingPath), std::runtime_error);
}

TEST(LoadFluidBinary, ThrowsOnCorruptFile) {
    const CorruptSnapshotFile file("atlas_corrupt_fluid_snapshot.bin");

    EXPECT_THROW(load_fluid_binary(file.path()), std::runtime_error);
}

TEST(LoadUniverseBinary, ThrowsOnCorruptFile) {
    const CorruptSnapshotFile file("atlas_corrupt_universe_snapshot.bin");

    EXPECT_THROW(load_universe_binary(file.path()), std::runtime_error);
}
