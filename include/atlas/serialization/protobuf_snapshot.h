#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>

#include <cstddef>
#include <optional>
#include <string_view>

namespace atlas {
template <typename T>
class Fluid;
}

namespace atlas {
template <typename T>
class Universe;
}

namespace atlas {

/**
 * @brief Serializable binary snapshot of a fluid instance.
 *
 * This structure stores core fluid metadata together with optional particle-state
 * buffers that may or may not be present in the serialized file.
 *
 * @tparam T Scalar type used by the snapshot.
 */
template <typename T>
struct FluidBinarySnapshot final {
    std::size_t buffer_size    = 0;
    std::size_t particle_count = 0;
    T statistical_weight       = T(1);

    HostBuffer<MaterialProperties<T>> properties;
    HostBuffer<GenerateOperator<T>> generators;

    std::optional<HostBuffer<Vector3<T>>> positions;
    std::optional<HostBuffer<Vector3<T>>> velocities;
    std::optional<HostBuffer<std::size_t>> species;
    std::optional<HostBuffer<int>> active;
    std::optional<HostBuffer<T>> temperature;
};

/**
 * @brief Serializable binary snapshot of a universe instance.
 *
 * This structure stores the geometric definition of the universe together with
 * optional field/state buffers that may or may not be present in the snapshot.
 *
 * @tparam T Scalar type used by the snapshot.
 */
template <typename T>
struct UniverseBinarySnapshot final {
    Vector3<T> lower_corner {};
    Vector3<T> upper_corner {};
    T cell_size = T(1);

    std::optional<HostBuffer<T>> temperature;
    std::optional<HostBuffer<Vector3<T>>> bulk_velocity;
    std::optional<HostBuffer<Vector3<T>>> field_force;
    std::optional<HostBuffer<T>> max_relative_speed;
    std::optional<HostBuffer<T>> thermal_energy;
    std::optional<HostBuffer<T>> number_particle;
    std::optional<HostBuffer<int>> collision_count;
    std::optional<HostBuffer<T>> knudsen_number;
};

/**
 * @brief Serialize a fluid instance into a protobuf-based binary snapshot.
 *
 * @tparam T Scalar type of the fluid.
 * @param fluid Fluid instance to serialize.
 * @param path Output file path.
 */
template <typename T>
void
save_fluid_binary(const atlas::Fluid<T>& fluid, std::string_view path);

/**
 * @brief Load a protobuf-based binary snapshot into a fluid snapshot structure.
 *
 * @tparam T Requested scalar type.
 * @param path Input file path.
 * @return Decoded fluid snapshot payload.
 */
template <typename T>
FluidBinarySnapshot<T>
load_fluid_binary(std::string_view path);

/**
 * @brief Serialize a universe instance into a protobuf-based binary snapshot.
 *
 * @tparam T Scalar type of the universe.
 * @param universe Universe instance to serialize.
 * @param path Output file path.
 */
template <typename T>
void
save_universe_binary(const atlas::Universe<T>& universe, std::string_view path);

/**
 * @brief Load a protobuf-based binary snapshot into a universe snapshot structure.
 *
 * @tparam T Requested scalar type.
 * @param path Input file path.
 * @return Decoded universe snapshot payload.
 */
template <typename T>
UniverseBinarySnapshot<T>
load_universe_binary(std::string_view path);

} // namespace atlas