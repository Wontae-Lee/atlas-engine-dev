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

/**
 * @brief Host-side, fully decoded contents of a fluid snapshot file.
 *
 * This is the plain-data result of parsing a serialized @ref atlas::Fluid: every
 * device column has already been copied back to the host as a @c HostBuffer, and the
 * protobuf wire types have been converted to the engine's own types. It carries no
 * device memory and no @ref atlas::Fluid object; @ref restore_fluid is what turns one
 * of these back into a live, device-resident fluid.
 *
 * A state buffer is stored in an @c std::optional so that a snapshot can faithfully
 * distinguish "this fluid never had that state attached" (@c std::nullopt) from "this
 * fluid had that state, and it happened to be empty". When present, every optional
 * buffer is sized to exactly @c buffer_size — @ref load_fluid_binary enforces that
 * invariant before returning.
 *
 * @see load_fluid_binary, restore_fluid, save_fluid_binary
 */
struct FluidBinarySnapshot final {
    std::size_t buffer_size    = 0;    ///< Allocated capacity of every column, in particles.
    std::size_t particle_count = 0;    ///< Live particle count; never exceeds @c buffer_size.
    float statistical_weight   = 1.0f; ///< Real molecules represented per simulated particle.

    /// Full species table, indexed by species id; empty when the fluid owned no dictionary.
    HostBuffer<Material> materials;

    std::optional<HostBuffer<Float3>> positions;           ///< Per-particle world positions, if present.
    std::optional<HostBuffer<Float3>> velocities;          ///< Per-particle velocities, if present.
    std::optional<HostBuffer<std::size_t>> species;        ///< Per-particle species ids, if present.
    std::optional<HostBuffer<int>> active;                 ///< Per-particle survivor flags (nonzero = alive).
    std::optional<HostBuffer<float>> temperature;          ///< Per-particle temperature, if present.
    std::optional<HostBuffer<float>> translational_energy; ///< Per-particle translational energy.
    std::optional<HostBuffer<float>> rotational_energy;    ///< Per-particle rotational energy.
    std::optional<HostBuffer<float>> vibrational_energy;   ///< Per-particle vibrational energy.
};

/**
 * @brief Host-side, fully decoded contents of a universe snapshot file.
 *
 * The grid geometry (@c lower_corner, @c upper_corner, @c cell_size) is always present;
 * each per-cell field is optional and set only when the serialized @ref atlas::Universe
 * carried the corresponding state. As with @ref FluidBinarySnapshot the buffers hold host
 * memory only; @ref restore_universe rebuilds a device-resident universe from them.
 *
 * @note Unlike the fluid snapshot, the universe fields are stored in the file as a
 *       @c repeated list keyed by an enum (see the @c UniverseStateKind proto enum), so
 *       @ref load_universe_binary populates these members by dispatching on that key.
 *
 * @see load_universe_binary, restore_universe, save_universe_binary
 */
struct UniverseBinarySnapshot final {
    Float3 lower_corner = Float3(0.0f, 0.0f, 0.0f); ///< Minimum corner of the domain, world units.
    Float3 upper_corner = Float3(0.0f, 0.0f, 0.0f); ///< Maximum corner of the domain, world units.
    float cell_size     = 1.0f;                     ///< Edge length of a cubic cell, world units.

    std::optional<HostBuffer<float>> temperature;        ///< Per-cell temperature, if present.
    std::optional<HostBuffer<Float3>> bulk_velocity;     ///< Per-cell bulk (mean) velocity, if present.
    std::optional<HostBuffer<Float3>> field_force;       ///< Per-cell external field force, if present.
    std::optional<HostBuffer<Float3>> gravity;           ///< Per-cell gravity vector, if present.
    std::optional<HostBuffer<float>> max_relative_speed; ///< Per-cell NTC max relative speed, if present.
    std::optional<HostBuffer<float>> max_sigma_g;        ///< Per-cell NTC max sigma*g, if present.
    std::optional<HostBuffer<float>> thermal_energy;     ///< Per-cell thermal energy, if present.
    std::optional<HostBuffer<float>> number_particle;    ///< Per-cell particle count estimate, if present.
    std::optional<HostBuffer<int>> collision_count;      ///< Per-cell collision tally, if present.
    std::optional<HostBuffer<float>> knudsen_number;     ///< Per-cell Knudsen number, if present.
    std::optional<HostBuffer<int>> allocated_solver;     ///< Per-cell selected-solver id, if present.
};

/**
 * @brief Serialize a live fluid to a protobuf snapshot file, overwriting any existing one.
 *
 * Copies each attached fluid state and the material dictionary from device memory back to
 * the host and writes them as a binary @c FluidSnapshot message. The @c active() flag column
 * is always written; the optional per-particle states are written only when attached.
 *
 * @param fluid The fluid to capture; read only, not modified.
 * @param path Filesystem path of the output file (truncated if it already exists).
 * @throws std::runtime_error if the fluid carries a state type this serializer does not
 *         know how to encode, or if the file cannot be opened or written.
 */
void
save_fluid_binary(const atlas::Fluid& fluid, std::string_view path);

/**
 * @brief Parse a fluid snapshot file into a host-side @ref FluidBinarySnapshot.
 *
 * Reads and decodes the file without touching the GPU: no @ref atlas::Fluid is constructed
 * and no device buffer is allocated. Validates the format version and the scalar type, then
 * checks that @c particle_count does not exceed @c buffer_size and that every present state
 * buffer is exactly @c buffer_size elements long.
 *
 * @param path Filesystem path of the snapshot file to read.
 * @return The decoded snapshot with all buffers resident on the host.
 * @throws std::runtime_error on a missing/corrupt file, a version or scalar-type mismatch,
 *         an unknown material type, or any of the size-consistency checks failing.
 */
FluidBinarySnapshot
load_fluid_binary(std::string_view path);

/**
 * @brief Serialize a live universe to a protobuf snapshot file, overwriting any existing one.
 *
 * Writes the grid geometry and every attached per-cell state (each copied from device memory
 * to the host) as a binary @c UniverseSnapshot message.
 *
 * @param universe The universe to capture; read only, not modified.
 * @param path Filesystem path of the output file (truncated if it already exists).
 * @throws std::runtime_error if the universe carries a state type this serializer does not
 *         know how to encode, or if the file cannot be opened or written.
 */
void
save_universe_binary(const atlas::Universe& universe, std::string_view path);

/**
 * @brief Parse a universe snapshot file into a host-side @ref UniverseBinarySnapshot.
 *
 * Reads and decodes the file without allocating any device memory. Validates the format
 * version and scalar type, then fills the optional per-cell buffers by dispatching on each
 * stored state's @c UniverseStateKind.
 *
 * @param path Filesystem path of the snapshot file to read.
 * @return The decoded snapshot with all buffers resident on the host.
 * @throws std::runtime_error on a missing/corrupt file, a version or scalar-type mismatch,
 *         a byte-size inconsistency, or an unrecognized universe state kind.
 */
UniverseBinarySnapshot
load_universe_binary(std::string_view path);

/**
 * @brief Load a fluid snapshot file and rebuild a live, device-resident fluid from it.
 *
 * Convenience wrapper over @ref load_fluid_binary followed by @ref atlas::Fluid::Builder:
 * it reconstructs the material dictionary (when the snapshot carried one), sizes the fluid
 * to the stored capacity and live count, and uploads each present state back to device
 * memory as the matching @c Fluid*State.
 *
 * @param path Filesystem path of the snapshot file to restore.
 * @return An owning host pointer to the reconstructed fluid.
 * @throws std::runtime_error for any failure surfaced by @ref load_fluid_binary or the builder.
 */
FluidHostPtr
restore_fluid(std::string_view path);

/**
 * @brief Load a universe snapshot file and rebuild a live, device-resident universe from it.
 *
 * Convenience wrapper over @ref load_universe_binary followed by @ref atlas::Universe::Builder:
 * it recreates the grid geometry and uploads each present per-cell buffer back to device memory
 * as the matching @c Universe*State.
 *
 * @param path Filesystem path of the snapshot file to restore.
 * @return An owning host pointer to the reconstructed universe.
 * @throws std::runtime_error for any failure surfaced by @ref load_universe_binary or the builder.
 */
UniverseHostPtr
restore_universe(std::string_view path);

}