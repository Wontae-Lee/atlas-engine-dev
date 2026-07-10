#include <atlas/serialization/protobuf_snapshot.h>

#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/material/atom.h>
#include <atlas/material/ion.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/material/molecule.h>
#include <atlas/material/neutron.h>
#include <atlas/material/solid.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include "atlas_snapshot.pb.h"

#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace atlas {
namespace {

    constexpr std::uint32_t kAtlasSnapshotVersion = 1u;

    constexpr atlas::proto::ScalarType kAtlasScalarType = atlas::proto::SCALAR_FLOAT32;

    void
    validate_scalar_type(const atlas::proto::ScalarType scalar_type,
                         const char* context) {
        if (scalar_type != kAtlasScalarType) {
            throw std::runtime_error(
                std::string(context)
                + ": scalar type in snapshot does not match the requested template type.");
        }
    }





    void
    set_vector3(atlas::proto::Vector3* target,
                const Float3& value) {
        target->set_x(static_cast<double>(value.x));
        target->set_y(static_cast<double>(value.y));
        target->set_z(static_cast<double>(value.z));
    }

    Float3
    read_vector3(const atlas::proto::Vector3& value) {
        return Float3(
            static_cast<float>(value.x()),
            static_cast<float>(value.y()),
            static_cast<float>(value.z()));
    }

    template <typename T>
    std::string
    pack_scalar_buffer(const HostBuffer<T>& values) {
        if (values.empty()) {
            return {};
        }

        return std::string(
            reinterpret_cast<const char*>(values.data()),
            values.size() * sizeof(T));
    }

    template <typename T>
    HostBuffer<T>
    unpack_scalar_buffer(const atlas::proto::RawBuffer& raw_buffer,
                         const char* context) {
        if (raw_buffer.element_count() == 0) {
            return {};
        }

        const auto expected_byte_size = static_cast<std::size_t>(raw_buffer.element_count()) * sizeof(T);

        if (raw_buffer.data().size() != expected_byte_size) {
            throw std::runtime_error(
                std::string(context)
                + ": raw buffer byte size does not match the encoded element count.");
        }

        HostBuffer<T> values(static_cast<std::size_t>(raw_buffer.element_count()));
        std::memcpy(values.data(), raw_buffer.data().data(), expected_byte_size);
        return values;
    }

    std::string
    pack_vector3_buffer(const HostBuffer<Float3>& values) {
        if (values.empty()) {
            return {};
        }

        std::string bytes;
        bytes.resize(values.size() * sizeof(Float3));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    }

    HostBuffer<Float3>
    unpack_vector3_buffer(const atlas::proto::RawBuffer& raw_buffer,
                          const char* context) {
        if (raw_buffer.element_count() == 0) {
            return {};
        }

        const auto expected_byte_size = static_cast<std::size_t>(raw_buffer.element_count()) * sizeof(Float3);

        if (raw_buffer.data().size() != expected_byte_size) {
            throw std::runtime_error(
                std::string(context)
                + ": vector buffer byte size does not match the encoded element count.");
        }

        HostBuffer<Float3> values(static_cast<std::size_t>(raw_buffer.element_count()));
        std::memcpy(values.data(), raw_buffer.data().data(), expected_byte_size);
        return values;
    }

    void
    set_material(atlas::proto::Material* target, const Material& source) {
        target->set_type(static_cast<int>(source.type));
        target->set_mass(static_cast<double>(source.mass()));
        target->set_translational_energy(static_cast<double>(source.translational_energy()));
        target->set_rotational_energy(static_cast<double>(source.rotational_energy()));
        target->set_vibrational_energy(static_cast<double>(source.vibrational_energy()));
        target->set_reference_diameter(static_cast<double>(source.reference_diameter()));
        target->set_reference_temperature(static_cast<double>(source.reference_temperature()));
        target->set_viscosity_index(static_cast<double>(source.viscosity_index()));
        target->set_scattering_parameter(static_cast<double>(source.scattering_parameter()));
    }

    Material
    read_material(const atlas::proto::Material& source) {
        const auto mass                 = static_cast<float>(source.mass());
        const auto translational_energy = static_cast<float>(source.translational_energy());
        const auto rotational_energy    = static_cast<float>(source.rotational_energy());
        const auto vibrational_energy   = static_cast<float>(source.vibrational_energy());
        const auto reference_diameter   = static_cast<float>(source.reference_diameter());
        const auto reference_temperature = static_cast<float>(source.reference_temperature());
        const auto viscosity_index      = static_cast<float>(source.viscosity_index());
        const auto scattering_parameter = static_cast<float>(source.scattering_parameter());

        switch (static_cast<MaterialType>(source.type())) {
        case MaterialType::molecule:
            return Material(Molecule(mass, translational_energy, rotational_energy, vibrational_energy,
                                     reference_diameter, reference_temperature, viscosity_index, scattering_parameter));
        case MaterialType::atom:
            return Material(Atom(mass, translational_energy, rotational_energy, vibrational_energy,
                                 reference_diameter, reference_temperature, viscosity_index, scattering_parameter));
        case MaterialType::ion:
            return Material(Ion(mass, translational_energy, rotational_energy, vibrational_energy,
                                reference_diameter, reference_temperature, viscosity_index, scattering_parameter));
        case MaterialType::neutron:
            return Material(Neutron(mass, translational_energy, rotational_energy, vibrational_energy,
                                    reference_diameter, reference_temperature, viscosity_index, scattering_parameter));
        case MaterialType::solid:
            // A solid carries only mass; everything else it reports is a constant.
            return Material(Solid(mass));
        }

        throw std::runtime_error("load_fluid_binary: unknown material type in snapshot.");
    }




    template <typename MessageT>
    void
    write_message(const MessageT& message,
                  std::string_view path,
                  const char* context) {
        std::ofstream output(std::string(path), std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(std::string(context) + ": failed to open output file.");
        }

        if (!message.SerializeToOstream(&output)) {
            throw std::runtime_error(std::string(context) + ": failed to serialize protobuf message.");
        }
    }

    template <typename MessageT>
    MessageT
    read_message(std::string_view path,
                 const char* context) {
        MessageT message;

        std::ifstream input(std::string(path), std::ios::binary);
        if (!input) {
            throw std::runtime_error(std::string(context) + ": failed to open snapshot file.");
        }

        if (!message.ParseFromIstream(&input)) {
            throw std::runtime_error(std::string(context) + ": failed to parse protobuf message.");
        }

        return message;
    }

    template <typename T>
    void
    set_raw_buffer(atlas::proto::RawBuffer* target,
                   const HostBuffer<T>& values) {
        target->set_element_count(values.size());
        target->set_data(pack_scalar_buffer(values));
    }

    void
    set_vector_raw_buffer(atlas::proto::RawBuffer* target,
                          const HostBuffer<Float3>& values) {
        target->set_element_count(values.size());
        target->set_data(pack_vector3_buffer(values));
    }

    void
    append_universe_scalar_state(atlas::proto::UniverseSnapshot* snapshot,
                                 const atlas::proto::UniverseStateKind kind,
                                 const HostBuffer<float>& values) {
        auto* state = snapshot->add_states();
        state->set_kind(kind);
        set_raw_buffer(state->mutable_scalar_buffer(), values);
    }

    void
    append_universe_vector_state(atlas::proto::UniverseSnapshot* snapshot,
                                 const atlas::proto::UniverseStateKind kind,
                                 const HostBuffer<Float3>& values) {
        auto* state = snapshot->add_states();
        state->set_kind(kind);
        set_vector_raw_buffer(state->mutable_vector_buffer(), values);
    }

    void
    append_universe_int_state(atlas::proto::UniverseSnapshot* snapshot,
                              const atlas::proto::UniverseStateKind kind,
                              const HostBuffer<int>& values) {
        auto* state = snapshot->add_states();
        state->set_kind(kind);
        set_raw_buffer(state->mutable_int_buffer(), values);
    }

} // namespace

void
save_fluid_binary(const atlas::Fluid& fluid, std::string_view path) {
    atlas::proto::FluidSnapshot snapshot;
    snapshot.set_version(kAtlasSnapshotVersion);
    snapshot.set_scalar_type(kAtlasScalarType);
    snapshot.set_buffer_size(fluid.buffer_size());
    snapshot.set_particle_count(fluid.particle_count());
    snapshot.set_statistical_weight(static_cast<double>(fluid.statistical_weight()));

    if (fluid.materials()) {
        const auto& device_materials = fluid.materials()->materials();
        const HostBuffer<Material> materials(device_materials.begin(), device_materials.end());

        for (const auto& material : materials) {
            set_material(snapshot.add_materials(), material);
        }
    }

    std::size_t known_state_count = 0;

    if (const auto* position_state = fluid.state<atlas::FluidPositionState>();
        position_state != nullptr) {
        const HostBuffer<Float3> positions(position_state->data().begin(), position_state->data().end());
        set_vector_raw_buffer(snapshot.mutable_positions(), positions);
        ++known_state_count;
    }

    if (const auto* velocity_state = fluid.state<atlas::FluidVelocityState>();
        velocity_state != nullptr) {
        const HostBuffer<Float3> velocities(velocity_state->data().begin(), velocity_state->data().end());
        set_vector_raw_buffer(snapshot.mutable_velocities(), velocities);
        ++known_state_count;
    }

    if (const auto* species_state = fluid.state<atlas::FluidSpeciesState>();
        species_state != nullptr) {
        const HostBuffer<std::size_t> species(species_state->data().begin(), species_state->data().end());
        set_raw_buffer(snapshot.mutable_species(), species);
        ++known_state_count;
    }

    {
        const HostBuffer<int> active(fluid.active().begin(), fluid.active().end());
        set_raw_buffer(snapshot.mutable_active(), active);
    }

    if (const auto* temperature_state = fluid.state<atlas::FluidTemperatureState>();
        temperature_state != nullptr) {
        const HostBuffer<float> temperature(temperature_state->data().begin(), temperature_state->data().end());
        set_raw_buffer(snapshot.mutable_temperature(), temperature);
        snapshot.set_includes_temperature_state(true);
        ++known_state_count;
    }

    if (const auto* state = fluid.state<atlas::FluidTranslationalEnergyState>();
        state != nullptr) {
        const HostBuffer<float> values(state->data().begin(), state->data().end());
        set_raw_buffer(snapshot.mutable_translational_energy(), values);
        snapshot.set_includes_translational_energy_state(true);
        ++known_state_count;
    }

    if (const auto* state = fluid.state<atlas::FluidRotationalEnergyState>();
        state != nullptr) {
        const HostBuffer<float> values(state->data().begin(), state->data().end());
        set_raw_buffer(snapshot.mutable_rotational_energy(), values);
        snapshot.set_includes_rotational_energy_state(true);
        ++known_state_count;
    }

    if (const auto* state = fluid.state<atlas::FluidVibrationalEnergyState>();
        state != nullptr) {
        const HostBuffer<float> values(state->data().begin(), state->data().end());
        set_raw_buffer(snapshot.mutable_vibrational_energy(), values);
        snapshot.set_includes_vibrational_energy_state(true);
        ++known_state_count;
    }

    if (fluid.states().size() != known_state_count) {
        throw std::runtime_error(
            "save_fluid_binary: encountered an unsupported fluid state type during protobuf serialization.");
    }

    write_message(snapshot, path, "save_fluid_binary");
}

FluidBinarySnapshot
load_fluid_binary(std::string_view path) {
    const auto snapshot = read_message<atlas::proto::FluidSnapshot>(path, "load_fluid_binary");

    if (snapshot.version() != kAtlasSnapshotVersion) {
        throw std::runtime_error("load_fluid_binary: unsupported fluid snapshot version.");
    }

    validate_scalar_type(snapshot.scalar_type(), "load_fluid_binary");

    FluidBinarySnapshot fluid_snapshot;
    fluid_snapshot.buffer_size        = static_cast<std::size_t>(snapshot.buffer_size());
    fluid_snapshot.particle_count     = static_cast<std::size_t>(snapshot.particle_count());
    fluid_snapshot.statistical_weight = static_cast<float>(snapshot.statistical_weight());

    fluid_snapshot.materials.reserve(snapshot.materials_size());
    for (const auto& material : snapshot.materials()) {
        fluid_snapshot.materials.push_back(read_material(material));
    }

    if (snapshot.positions().element_count() > 0) {
        fluid_snapshot.positions = unpack_vector3_buffer(snapshot.positions(), "load_fluid_binary/positions");
    }

    if (snapshot.velocities().element_count() > 0) {
        fluid_snapshot.velocities = unpack_vector3_buffer(snapshot.velocities(), "load_fluid_binary/velocities");
    }

    if (snapshot.species().element_count() > 0) {
        fluid_snapshot.species = unpack_scalar_buffer<std::size_t>(snapshot.species(), "load_fluid_binary/species");
    }

    if (snapshot.active().element_count() > 0) {
        fluid_snapshot.active = unpack_scalar_buffer<int>(snapshot.active(), "load_fluid_binary/active");
    }

    if (snapshot.includes_temperature_state()) {
        fluid_snapshot.temperature = unpack_scalar_buffer<float>(snapshot.temperature(), "load_fluid_binary/temperature");
    }

    if (snapshot.includes_translational_energy_state()) {
        fluid_snapshot.translational_energy = unpack_scalar_buffer<float>(snapshot.translational_energy(), "load_fluid_binary/translational_energy");
    }

    if (snapshot.includes_rotational_energy_state()) {
        fluid_snapshot.rotational_energy = unpack_scalar_buffer<float>(snapshot.rotational_energy(), "load_fluid_binary/rotational_energy");
    }

    if (snapshot.includes_vibrational_energy_state()) {
        fluid_snapshot.vibrational_energy = unpack_scalar_buffer<float>(snapshot.vibrational_energy(), "load_fluid_binary/vibrational_energy");
    }

    if (fluid_snapshot.particle_count > fluid_snapshot.buffer_size) {
        throw std::runtime_error(
            "load_fluid_binary: particle_count exceeds buffer_size in snapshot.");
    }

    const auto validate_state_size = [&](const auto& state_buffer, const char* state_name) {
        if (state_buffer.has_value() && state_buffer->size() != fluid_snapshot.buffer_size) {
            throw std::runtime_error(
                std::string("load_fluid_binary: ")
                + state_name + " state size does not match buffer_size.");
        }
    };

    validate_state_size(fluid_snapshot.positions, "position");
    validate_state_size(fluid_snapshot.velocities, "velocity");
    validate_state_size(fluid_snapshot.species, "species");
    validate_state_size(fluid_snapshot.active, "active");
    validate_state_size(fluid_snapshot.temperature, "temperature");
    validate_state_size(fluid_snapshot.translational_energy, "translational energy");
    validate_state_size(fluid_snapshot.rotational_energy, "rotational energy");
    validate_state_size(fluid_snapshot.vibrational_energy, "vibrational energy");

    return fluid_snapshot;
}

void
save_universe_binary(const atlas::Universe& universe, std::string_view path) {
    atlas::proto::UniverseSnapshot snapshot;
    snapshot.set_version(kAtlasSnapshotVersion);
    snapshot.set_scalar_type(kAtlasScalarType);
    set_vector3(snapshot.mutable_lower_corner(), universe.lower_corner());
    set_vector3(snapshot.mutable_upper_corner(), universe.upper_corner());
    snapshot.set_cell_size(static_cast<double>(universe.cell_size()));

    std::size_t known_state_count = 0;

    if (const auto* temperature_state = universe.state<atlas::UniverseTemperatureState>();
        temperature_state != nullptr) {
        append_universe_scalar_state(
            &snapshot,
            atlas::proto::UNIVERSE_TEMPERATURE,
            HostBuffer<float>(temperature_state->data().begin(), temperature_state->data().end()));
        ++known_state_count;
    }

    if (const auto* bulk_velocity_state = universe.state<atlas::UniverseBulkVelocityState>();
        bulk_velocity_state != nullptr) {
        append_universe_vector_state(
            &snapshot,
            atlas::proto::UNIVERSE_BULK_VELOCITY,
            HostBuffer<Float3>(bulk_velocity_state->data().begin(), bulk_velocity_state->data().end()));
        ++known_state_count;
    }

    if (const auto* field_force_state = universe.state<atlas::UniverseFieldForceState>();
        field_force_state != nullptr) {
        append_universe_vector_state(
            &snapshot,
            atlas::proto::UNIVERSE_FIELD_FORCE,
            HostBuffer<Float3>(field_force_state->data().begin(), field_force_state->data().end()));
        ++known_state_count;
    }

    if (const auto* gravity_state = universe.state<atlas::UniverseGravityState>();
        gravity_state != nullptr) {
        append_universe_vector_state(
            &snapshot,
            atlas::proto::UNIVERSE_GRAVITY,
            HostBuffer<Float3>(gravity_state->data().begin(), gravity_state->data().end()));
        ++known_state_count;
    }

    if (const auto* max_sigma_g_state = universe.state<atlas::UniverseMaxSigmaGState>();
        max_sigma_g_state != nullptr) {
        append_universe_scalar_state(
            &snapshot,
            atlas::proto::UNIVERSE_MAX_SIGMA_G,
            HostBuffer<float>(max_sigma_g_state->data().begin(), max_sigma_g_state->data().end()));
        ++known_state_count;
    }

    if (const auto* allocated_solver_state = universe.state<atlas::UniverseAllocatedSolverState>();
        allocated_solver_state != nullptr) {
        append_universe_int_state(
            &snapshot,
            atlas::proto::UNIVERSE_ALLOCATED_SOLVER,
            HostBuffer<int>(allocated_solver_state->data().begin(), allocated_solver_state->data().end()));
        ++known_state_count;
    }

    if (const auto* max_relative_speed_state = universe.state<atlas::UniverseMaxRelativeSpeedState>();
        max_relative_speed_state != nullptr) {
        append_universe_scalar_state(
            &snapshot,
            atlas::proto::UNIVERSE_MAX_RELATIVE_SPEED,
            HostBuffer<float>(max_relative_speed_state->data().begin(), max_relative_speed_state->data().end()));
        ++known_state_count;
    }

    if (const auto* thermal_energy_state = universe.state<atlas::UniverseThermalEnergyState>();
        thermal_energy_state != nullptr) {
        append_universe_scalar_state(
            &snapshot,
            atlas::proto::UNIVERSE_THERMAL_ENERGY,
            HostBuffer<float>(thermal_energy_state->data().begin(), thermal_energy_state->data().end()));
        ++known_state_count;
    }

    if (const auto* number_particle_state = universe.state<atlas::UniverseNumberParticleState>();
        number_particle_state != nullptr) {
        append_universe_scalar_state(
            &snapshot,
            atlas::proto::UNIVERSE_NUMBER_PARTICLE,
            HostBuffer<float>(number_particle_state->data().begin(), number_particle_state->data().end()));
        ++known_state_count;
    }

    if (const auto* collision_count_state = universe.state<atlas::UniverseCollisionCountState>();
        collision_count_state != nullptr) {
        append_universe_int_state(
            &snapshot,
            atlas::proto::UNIVERSE_COLLISION_COUNT,
            HostBuffer<int>(collision_count_state->data().begin(), collision_count_state->data().end()));
        ++known_state_count;
    }

    if (const auto* knudsen_number_state = universe.state<atlas::UniverseKnudsenNumberState>();
        knudsen_number_state != nullptr) {
        append_universe_scalar_state(
            &snapshot,
            atlas::proto::UNIVERSE_KNUDSEN_NUMBER,
            HostBuffer<float>(knudsen_number_state->data().begin(), knudsen_number_state->data().end()));
        ++known_state_count;
    }

    if (universe.states().size() != known_state_count) {
        throw std::runtime_error(
            "save_universe_binary: encountered an unsupported universe state type during protobuf serialization.");
    }

    write_message(snapshot, path, "save_universe_binary");
}

UniverseBinarySnapshot
load_universe_binary(std::string_view path) {
    const auto snapshot = read_message<atlas::proto::UniverseSnapshot>(path, "load_universe_binary");

    if (snapshot.version() != kAtlasSnapshotVersion) {
        throw std::runtime_error("load_universe_binary: unsupported universe snapshot version.");
    }

    validate_scalar_type(snapshot.scalar_type(), "load_universe_binary");

    UniverseBinarySnapshot universe_snapshot;
    universe_snapshot.lower_corner = read_vector3(snapshot.lower_corner());
    universe_snapshot.upper_corner = read_vector3(snapshot.upper_corner());
    universe_snapshot.cell_size    = static_cast<float>(snapshot.cell_size());

    for (const auto& state : snapshot.states()) {
        switch (state.kind()) {
        case atlas::proto::UNIVERSE_TEMPERATURE:
            universe_snapshot.temperature = unpack_scalar_buffer<float>(state.scalar_buffer(), "load_universe_binary/temperature");
            break;
        case atlas::proto::UNIVERSE_BULK_VELOCITY:
            universe_snapshot.bulk_velocity = unpack_vector3_buffer(state.vector_buffer(), "load_universe_binary/bulk_velocity");
            break;
        case atlas::proto::UNIVERSE_FIELD_FORCE:
            universe_snapshot.field_force = unpack_vector3_buffer(state.vector_buffer(), "load_universe_binary/field_force");
            break;
        case atlas::proto::UNIVERSE_MAX_RELATIVE_SPEED:
            universe_snapshot.max_relative_speed = unpack_scalar_buffer<float>(state.scalar_buffer(), "load_universe_binary/max_relative_speed");
            break;
        case atlas::proto::UNIVERSE_THERMAL_ENERGY:
            universe_snapshot.thermal_energy = unpack_scalar_buffer<float>(state.scalar_buffer(), "load_universe_binary/thermal_energy");
            break;
        case atlas::proto::UNIVERSE_NUMBER_PARTICLE:
            universe_snapshot.number_particle = unpack_scalar_buffer<float>(state.scalar_buffer(), "load_universe_binary/number_particle");
            break;
        case atlas::proto::UNIVERSE_COLLISION_COUNT:
            universe_snapshot.collision_count = unpack_scalar_buffer<int>(state.int_buffer(), "load_universe_binary/collision_count");
            break;
        case atlas::proto::UNIVERSE_KNUDSEN_NUMBER:
            universe_snapshot.knudsen_number = unpack_scalar_buffer<float>(state.scalar_buffer(), "load_universe_binary/knudsen_number");
            break;
        case atlas::proto::UNIVERSE_GRAVITY:
            universe_snapshot.gravity = unpack_vector3_buffer(state.vector_buffer(), "load_universe_binary/gravity");
            break;
        case atlas::proto::UNIVERSE_MAX_SIGMA_G:
            universe_snapshot.max_sigma_g = unpack_scalar_buffer<float>(state.scalar_buffer(), "load_universe_binary/max_sigma_g");
            break;
        case atlas::proto::UNIVERSE_ALLOCATED_SOLVER:
            universe_snapshot.allocated_solver = unpack_scalar_buffer<int>(state.int_buffer(), "load_universe_binary/allocated_solver");
            break;
        default:
            throw std::runtime_error(
                "load_universe_binary: encountered an unknown universe state kind.");
        }
    }

    return universe_snapshot;
}

FluidHostPtr
restore_fluid(const std::string_view path) {
    const auto snapshot = load_fluid_binary(path);

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

} // namespace atlas
