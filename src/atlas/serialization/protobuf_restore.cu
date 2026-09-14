#include <atlas/serialization/protobuf_snapshot.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/universe/universe_state.h>

#include <memory>
#include <string_view>

/**
 * @file
 * @brief Upload decoded snapshots without including protobuf in a device-compiled file.
 *
 * Device-buffer construction and destruction require nvcc for the CUDA backend.
 * The protobuf parser stays in protobuf_snapshot.cpp, which the host compiler builds.
 */

namespace atlas {

/**
 * @brief Restore particle columns and the material dictionary from a decoded snapshot.
 * @param path Binary snapshot file to load.
 * @return A device-resident fluid owning the restored buffers.
 * @throws std::runtime_error if loading the snapshot or rebuilding the fluid fails.
 */
FluidHostPtr
restore_fluid(const std::string_view path) {
    const auto snapshot = load_fluid_binary(path);

    // Leave the dictionary null when the snapshot carried no materials; the fluid builder
    // accepts a null dictionary, matching a fluid that was saved without one.
    MaterialDictionaryHostPtr materials;

    if (!snapshot.materials.empty()) {
        materials = atlas::make_host_shared<MaterialDictionary>(
            MaterialDictionary::builder().with_materials(snapshot.materials).build());
    }

    auto fluid = Fluid::builder()
                     .with_buffer_size(snapshot.buffer_size)
                     .with_particle_count(snapshot.particle_count)
                     .with_statistical_weight(snapshot.statistical_weight)
                     .with_materials(materials)
                     .make_host_unique();

    if (snapshot.positions.has_value()) {
        fluid->set_state<FluidPositionState>(std::make_unique<FluidPositionState>(
            DeviceBuffer<Float3>(snapshot.positions->begin(), snapshot.positions->end())));
    }

    if (snapshot.velocities.has_value()) {
        fluid->set_state<FluidVelocityState>(std::make_unique<FluidVelocityState>(
            DeviceBuffer<Float3>(snapshot.velocities->begin(), snapshot.velocities->end())));
    }

    if (snapshot.species.has_value()) {
        fluid->set_state<FluidSpeciesState>(std::make_unique<FluidSpeciesState>(
            DeviceBuffer<std::size_t>(snapshot.species->begin(), snapshot.species->end())));
    }

    if (snapshot.active.has_value()) {
        // active is a built-in column, not a pluggable state, so upload it by assigning the
        // member buffer directly rather than through set_state<>.
        fluid->active() = DeviceBuffer<int>(snapshot.active->begin(), snapshot.active->end());
    }

    if (snapshot.temperature.has_value()) {
        fluid->set_state<FluidTemperatureState>(std::make_unique<FluidTemperatureState>(
            DeviceBuffer<float>(snapshot.temperature->begin(), snapshot.temperature->end())));
    }

    if (snapshot.translational_energy.has_value()) {
        fluid->set_state<FluidTranslationalEnergyState>(std::make_unique<FluidTranslationalEnergyState>(
            DeviceBuffer<float>(snapshot.translational_energy->begin(), snapshot.translational_energy->end())));
    }

    if (snapshot.rotational_energy.has_value()) {
        fluid->set_state<FluidRotationalEnergyState>(std::make_unique<FluidRotationalEnergyState>(
            DeviceBuffer<float>(snapshot.rotational_energy->begin(), snapshot.rotational_energy->end())));
    }

    if (snapshot.vibrational_energy.has_value()) {
        fluid->set_state<FluidVibrationalEnergyState>(std::make_unique<FluidVibrationalEnergyState>(
            DeviceBuffer<float>(snapshot.vibrational_energy->begin(), snapshot.vibrational_energy->end())));
    }

    return fluid;
}

/**
 * @brief Restore grid geometry and every recorded per-cell field from a snapshot.
 * @param path Binary snapshot file to load.
 * @return A device-resident universe owning the restored fields.
 * @throws std::runtime_error if loading the snapshot or rebuilding the universe fails.
 */
UniverseHostPtr
restore_universe(const std::string_view path) {
    const auto snapshot = load_universe_binary(path);

    auto universe = Universe::builder()
                        .with_lower_corner(snapshot.lower_corner)
                        .with_upper_corner(snapshot.upper_corner)
                        .with_cell_size(snapshot.cell_size)
                        .make_host_unique();

    if (snapshot.temperature.has_value()) {
        universe->set_state<UniverseTemperatureState>(std::make_unique<UniverseTemperatureState>(
            DeviceBuffer<float>(snapshot.temperature->begin(), snapshot.temperature->end())));
    }

    if (snapshot.bulk_velocity.has_value()) {
        universe->set_state<UniverseBulkVelocityState>(std::make_unique<UniverseBulkVelocityState>(
            DeviceBuffer<Float3>(snapshot.bulk_velocity->begin(), snapshot.bulk_velocity->end())));
    }

    if (snapshot.field_force.has_value()) {
        universe->set_state<UniverseFieldForceState>(std::make_unique<UniverseFieldForceState>(
            DeviceBuffer<Float3>(snapshot.field_force->begin(), snapshot.field_force->end())));
    }

    if (snapshot.gravity.has_value()) {
        universe->set_state<UniverseGravityState>(std::make_unique<UniverseGravityState>(
            DeviceBuffer<Float3>(snapshot.gravity->begin(), snapshot.gravity->end())));
    }

    if (snapshot.max_relative_speed.has_value()) {
        universe->set_state<UniverseMaxRelativeSpeedState>(std::make_unique<UniverseMaxRelativeSpeedState>(
            DeviceBuffer<float>(snapshot.max_relative_speed->begin(), snapshot.max_relative_speed->end())));
    }

    if (snapshot.max_sigma_g.has_value()) {
        universe->set_state<UniverseMaxSigmaGState>(std::make_unique<UniverseMaxSigmaGState>(
            DeviceBuffer<float>(snapshot.max_sigma_g->begin(), snapshot.max_sigma_g->end())));
    }

    if (snapshot.thermal_energy.has_value()) {
        universe->set_state<UniverseThermalEnergyState>(std::make_unique<UniverseThermalEnergyState>(
            DeviceBuffer<float>(snapshot.thermal_energy->begin(), snapshot.thermal_energy->end())));
    }

    if (snapshot.number_particle.has_value()) {
        universe->set_state<UniverseNumberParticleState>(std::make_unique<UniverseNumberParticleState>(
            DeviceBuffer<float>(snapshot.number_particle->begin(), snapshot.number_particle->end())));
    }

    if (snapshot.collision_count.has_value()) {
        universe->set_state<UniverseCollisionCountState>(std::make_unique<UniverseCollisionCountState>(
            DeviceBuffer<int>(snapshot.collision_count->begin(), snapshot.collision_count->end())));
    }

    if (snapshot.knudsen_number.has_value()) {
        universe->set_state<UniverseKnudsenNumberState>(std::make_unique<UniverseKnudsenNumberState>(
            DeviceBuffer<float>(snapshot.knudsen_number->begin(), snapshot.knudsen_number->end())));
    }

    if (snapshot.allocated_solver.has_value()) {
        universe->set_state<UniverseAllocatedSolverState>(std::make_unique<UniverseAllocatedSolverState>(
            DeviceBuffer<int>(snapshot.allocated_solver->begin(), snapshot.allocated_solver->end())));
    }

    return universe;
}

}
