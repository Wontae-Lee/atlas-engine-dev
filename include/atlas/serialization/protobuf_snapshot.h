#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/material/material.h>
#include <atlas/math/math.h>
#include <atlas/universe/universe.h>

#include <cstddef>
#include <optional>
#include <string_view>

namespace atlas {

// The fluid's own data, read back from a snapshot. Sources, generators,
// colliders and sinks belong to the system that drives the fluid, not to the
// fluid, so none of them appear here.
struct FluidBinarySnapshot final {
    std::size_t buffer_size    = 0;
    std::size_t particle_count = 0;
    float statistical_weight   = 1.0f;

    // Indexed by species id.
    HostBuffer<Material> materials;

    std::optional<HostBuffer<Float3>> positions;
    std::optional<HostBuffer<Float3>> velocities;
    std::optional<HostBuffer<std::size_t>> species;
    std::optional<HostBuffer<int>> active;
    std::optional<HostBuffer<float>> temperature;
    std::optional<HostBuffer<float>> translational_energy;
    std::optional<HostBuffer<float>> rotational_energy;
    std::optional<HostBuffer<float>> vibrational_energy;
};

struct UniverseBinarySnapshot final {
    Float3 lower_corner = Float3(0.0f, 0.0f, 0.0f);
    Float3 upper_corner = Float3(0.0f, 0.0f, 0.0f);
    float cell_size     = 1.0f;

    std::optional<HostBuffer<float>> temperature;
    std::optional<HostBuffer<Float3>> bulk_velocity;
    std::optional<HostBuffer<Float3>> field_force;
    std::optional<HostBuffer<Float3>> gravity;
    std::optional<HostBuffer<float>> max_relative_speed;
    std::optional<HostBuffer<float>> max_sigma_g;
    std::optional<HostBuffer<float>> thermal_energy;
    std::optional<HostBuffer<float>> number_particle;
    std::optional<HostBuffer<int>> collision_count;
    std::optional<HostBuffer<float>> knudsen_number;
    std::optional<HostBuffer<int>> allocated_solver;
};

void
save_fluid_binary(const atlas::Fluid& fluid, std::string_view path);

FluidBinarySnapshot
load_fluid_binary(std::string_view path);

void
save_universe_binary(const atlas::Universe& universe, std::string_view path);

UniverseBinarySnapshot
load_universe_binary(std::string_view path);

// Rebuilds the object a snapshot was taken of, restoring its material
// dictionary and every state the snapshot carries. Together these are what a
// restart needs: a fluid and a universe that resume where the run stopped.
FluidHostPtr
restore_fluid(std::string_view path);

UniverseHostPtr
restore_universe(std::string_view path);

}
