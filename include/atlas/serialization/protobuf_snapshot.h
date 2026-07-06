#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/generator/generate.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>

#include <cstddef>
#include <optional>
#include <string_view>

namespace atlas {
class Fluid;
}

namespace atlas {
class Universe;
}

namespace atlas {

struct FluidBinarySnapshot final {
    std::size_t buffer_size    = 0;
    std::size_t particle_count = 0;
    float statistical_weight   = 1.0f;

    HostBuffer<MaterialProperties> properties;
    HostBuffer<Generate> generators;

    std::optional<HostBuffer<Float3>> positions;
    std::optional<HostBuffer<Float3>> velocities;
    std::optional<HostBuffer<std::size_t>> species;
    std::optional<HostBuffer<int>> active;
    std::optional<HostBuffer<float>> temperature;
};

struct UniverseBinarySnapshot final {
    Float3 lower_corner = Float3(0.0f, 0.0f, 0.0f);
    Float3 upper_corner = Float3(0.0f, 0.0f, 0.0f);
    float cell_size     = 1.0f;

    std::optional<HostBuffer<float>> temperature;
    std::optional<HostBuffer<Float3>> bulk_velocity;
    std::optional<HostBuffer<Float3>> field_force;
    std::optional<HostBuffer<float>> max_relative_speed;
    std::optional<HostBuffer<float>> thermal_energy;
    std::optional<HostBuffer<float>> number_particle;
    std::optional<HostBuffer<int>> collision_count;
    std::optional<HostBuffer<float>> knudsen_number;
};

void
save_fluid_binary(const atlas::Fluid& fluid, std::string_view path);

FluidBinarySnapshot
load_fluid_binary(std::string_view path);

void
save_universe_binary(const atlas::Universe& universe, std::string_view path);

UniverseBinarySnapshot
load_universe_binary(std::string_view path);

}
