#include <atlas/serialization/protobuf_snapshot.h>

#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include "atlas_snapshot.pb.h"

#include <cstring>
#include <fstream>
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
    set_optional_double(atlas::proto::OptionalDouble* target,
                        const std::optional<float>& value) {
        target->set_has_value(value.has_value());
        if (value.has_value()) {
            target->set_value(static_cast<double>(*value));
        }
    }

    void
    set_optional_int32(atlas::proto::OptionalInt32* target,
                       const std::optional<int>& value) {
        target->set_has_value(value.has_value());
        if (value.has_value()) {
            target->set_value(*value);
        }
    }

    std::optional<float>
    read_optional_double(const atlas::proto::OptionalDouble& value) {
        if (!value.has_value()) {
            return std::nullopt;
        }

        return static_cast<float>(value.value());
    }

    std::optional<int>
    read_optional_int32(const atlas::proto::OptionalInt32& value) {
        if (!value.has_value()) {
            return std::nullopt;
        }

        return value.value();
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

        if (raw_buffer.data().size() != static_cast<int>(expected_byte_size)) {
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

        if (raw_buffer.data().size() != static_cast<int>(expected_byte_size)) {
            throw std::runtime_error(
                std::string(context)
                + ": vector buffer byte size does not match the encoded element count.");
        }

        HostBuffer<Float3> values(static_cast<std::size_t>(raw_buffer.element_count()));
        std::memcpy(values.data(), raw_buffer.data().data(), expected_byte_size);
        return values;
    }

    void
    set_material_property(atlas::proto::MaterialProperty* target,
                          const MaterialProperties& source) {
        target->set_type(static_cast<int>(source.type));
        target->set_mass(static_cast<double>(source.mass));
        target->set_molecular_mass(static_cast<double>(source.molecular_mass));
        set_optional_double(target->mutable_translational_energy(), source.translational_energy);
        set_optional_double(target->mutable_rotational_energy(), source.rotational_energy);
        set_optional_double(target->mutable_vibrational_energy(), source.vibrational_energy);
        set_optional_int32(target->mutable_species_id(), source.species_id);
        set_optional_double(target->mutable_reference_diameter(), source.reference_diameter);
        set_optional_double(target->mutable_reference_temperature(), source.reference_temperature);
        set_optional_double(target->mutable_viscosity_index(), source.viscosity_index);
        set_optional_double(target->mutable_scattering_parameter(), source.scattering_parameter);
        set_optional_double(target->mutable_rest_density(), source.rest_density);
        set_optional_double(target->mutable_pressure_coefficient(), source.pressure_coefficient);
        set_optional_double(target->mutable_dynamic_viscosity(), source.dynamic_viscosity);
        set_optional_double(target->mutable_electronic_energy(), source.electronic_energy);
        set_optional_int32(target->mutable_charge(), source.charge);
    }

    MaterialProperties
    read_material_property(const atlas::proto::MaterialProperty& source) {
        MaterialProperties material {};
        material.type                 = static_cast<atlas::MaterialType::Value>(source.type());
        material.mass                 = static_cast<float>(source.mass());
        material.molecular_mass       = static_cast<float>(source.molecular_mass());
        material.translational_energy = read_optional_double(source.translational_energy());
        material.rotational_energy    = read_optional_double(source.rotational_energy());
        material.vibrational_energy   = read_optional_double(source.vibrational_energy());
        material.species_id           = read_optional_int32(source.species_id());
        material.reference_diameter    = read_optional_double(source.reference_diameter());
        material.reference_temperature = read_optional_double(source.reference_temperature());
        material.viscosity_index       = read_optional_double(source.viscosity_index());
        material.scattering_parameter  = read_optional_double(source.scattering_parameter());
        material.rest_density          = read_optional_double(source.rest_density());
        material.pressure_coefficient  = read_optional_double(source.pressure_coefficient());
        material.dynamic_viscosity     = read_optional_double(source.dynamic_viscosity());
        material.electronic_energy     = read_optional_double(source.electronic_energy());
        material.charge                = read_optional_int32(source.charge());
        return material;
    }

    atlas::proto::GeneratorType
    to_proto_generator_type(const GenerateType type) {
        switch (type) {
        case GenerateType::maxwell_sigma:
            return atlas::proto::GENERATOR_MAXWELL_SIGMA;
        case GenerateType::maxwell_boltzmann:
            return atlas::proto::GENERATOR_MAXWELL_BOLTZMANN;
        case GenerateType::uniform:
            return atlas::proto::GENERATOR_UNIFORM;
        default:
            throw std::runtime_error("Unsupported generator operator type.");
        }
    }

    Generate
    read_generator(const atlas::proto::GeneratorOperator& source) {
        switch (source.type()) {
        case atlas::proto::GENERATOR_UNIFORM:
            return Generate(UniformGenerate(source.seed()));
        case atlas::proto::GENERATOR_MAXWELL_SIGMA:
            return Generate(MaxwellSigmaGenerate(source.seed()));
        case atlas::proto::GENERATOR_MAXWELL_BOLTZMANN:
            return Generate(
                MaxwellBoltzmannGenerate(
                    source.seed(),
                    read_vector3(source.bulk_velocity())));
        default:
            throw std::runtime_error("Fluid snapshot contains an unknown generator type.");
        }
    }

    void
    set_generator(atlas::proto::GeneratorOperator* target,
                  const Generate& source) {
        target->set_type(to_proto_generator_type(source.type));

        switch (source.type) {
        case GenerateType::uniform:
            target->set_seed(source.uniform.seed);
            break;
        case GenerateType::maxwell_sigma:
            target->set_seed(source.maxwell_sigma.seed);
            break;
        case GenerateType::maxwell_boltzmann:
            target->set_seed(source.maxwell_boltzmann.seed);
            set_vector3(target->mutable_bulk_velocity(), source.maxwell_boltzmann.bulk_velocity);
            break;
        default:
            throw std::runtime_error("Unsupported generator operator type.");
        }
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

    {
        HostBuffer<MaterialProperties> material_properties(
            fluid.particle_properties().begin(),
            fluid.particle_properties().end());

        for (const auto& material_property : material_properties) {
            set_material_property(snapshot.add_properties(), material_property);
        }
    }

    {
        HostBuffer<Generate> generator_operators(
            fluid.generators().begin(),
            fluid.generators().end());

        for (const auto& generator_operator : generator_operators) {
            set_generator(snapshot.add_generators(), generator_operator);
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

    if (const auto* active_state = fluid.state<atlas::FluidActiveState>();
        active_state != nullptr) {
        const HostBuffer<int> active(active_state->data().begin(), active_state->data().end());
        set_raw_buffer(snapshot.mutable_active(), active);
        ++known_state_count;
    }

    if (const auto* temperature_state = fluid.state<atlas::FluidTemperatureState>();
        temperature_state != nullptr) {
        const HostBuffer<float> temperature(temperature_state->data().begin(), temperature_state->data().end());
        set_raw_buffer(snapshot.mutable_temperature(), temperature);
        snapshot.set_includes_temperature_state(true);
        ++known_state_count;
    }

    if (fluid.states().size() != known_state_count) {
        throw std::runtime_error(
            "Fluid::save: encountered an unsupported fluid state type during protobuf serialization.");
    }

    write_message(snapshot, path, "Fluid::save");
}

FluidBinarySnapshot
load_fluid_binary(std::string_view path) {
    const auto snapshot = read_message<atlas::proto::FluidSnapshot>(path, "Fluid::Builder::with_binary");

    if (snapshot.version() != kAtlasSnapshotVersion) {
        throw std::runtime_error("Fluid::Builder::with_binary: unsupported fluid snapshot version.");
    }

    validate_scalar_type(snapshot.scalar_type(), "Fluid::Builder::with_binary");

    FluidBinarySnapshot fluid_snapshot;
    fluid_snapshot.buffer_size        = static_cast<std::size_t>(snapshot.buffer_size());
    fluid_snapshot.particle_count     = static_cast<std::size_t>(snapshot.particle_count());
    fluid_snapshot.statistical_weight = static_cast<float>(snapshot.statistical_weight());

    fluid_snapshot.properties.reserve(snapshot.properties_size());
    for (const auto& property : snapshot.properties()) {
        fluid_snapshot.properties.push_back(read_material_property(property));
    }

    fluid_snapshot.generators.reserve(snapshot.generators_size());
    for (const auto& generator : snapshot.generators()) {
        fluid_snapshot.generators.push_back(read_generator(generator));
    }

    if (snapshot.positions().element_count() > 0) {
        fluid_snapshot.positions = unpack_vector3_buffer(snapshot.positions(), "Fluid::Builder::with_binary/positions");
    }

    if (snapshot.velocities().element_count() > 0) {
        fluid_snapshot.velocities = unpack_vector3_buffer(snapshot.velocities(), "Fluid::Builder::with_binary/velocities");
    }

    if (snapshot.species().element_count() > 0) {
        fluid_snapshot.species = unpack_scalar_buffer<std::size_t>(snapshot.species(), "Fluid::Builder::with_binary/species");
    }

    if (snapshot.active().element_count() > 0) {
        fluid_snapshot.active = unpack_scalar_buffer<int>(snapshot.active(), "Fluid::Builder::with_binary/active");
    }

    if (snapshot.includes_temperature_state()) {
        fluid_snapshot.temperature = unpack_scalar_buffer<float>(snapshot.temperature(), "Fluid::Builder::with_binary/temperature");
    }

    if (fluid_snapshot.particle_count > fluid_snapshot.buffer_size) {
        throw std::runtime_error(
            "Fluid::Builder::with_binary: particle_count exceeds buffer_size in snapshot.");
    }

    const auto validate_state_size = [&](const auto& state_buffer, const char* state_name) {
        if (state_buffer.has_value() && state_buffer->size() != fluid_snapshot.buffer_size) {
            throw std::runtime_error(
                std::string("Fluid::Builder::with_binary: ")
                + state_name + " state size does not match buffer_size.");
        }
    };

    validate_state_size(fluid_snapshot.positions, "position");
    validate_state_size(fluid_snapshot.velocities, "velocity");
    validate_state_size(fluid_snapshot.species, "species");
    validate_state_size(fluid_snapshot.active, "active");
    validate_state_size(fluid_snapshot.temperature, "temperature");

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
            "Universe::save: encountered an unsupported universe state type during protobuf serialization.");
    }

    write_message(snapshot, path, "Universe::save");
}

UniverseBinarySnapshot
load_universe_binary(std::string_view path) {
    const auto snapshot = read_message<atlas::proto::UniverseSnapshot>(path, "Universe::Builder::with_binary");

    if (snapshot.version() != kAtlasSnapshotVersion) {
        throw std::runtime_error("Universe::Builder::with_binary: unsupported universe snapshot version.");
    }

    validate_scalar_type(snapshot.scalar_type(), "Universe::Builder::with_binary");

    UniverseBinarySnapshot universe_snapshot;
    universe_snapshot.lower_corner = read_vector3(snapshot.lower_corner());
    universe_snapshot.upper_corner = read_vector3(snapshot.upper_corner());
    universe_snapshot.cell_size    = static_cast<float>(snapshot.cell_size());

    for (const auto& state : snapshot.states()) {
        switch (state.kind()) {
        case atlas::proto::UNIVERSE_TEMPERATURE:
            universe_snapshot.temperature = unpack_scalar_buffer<float>(state.scalar_buffer(), "Universe::Builder::with_binary/temperature");
            break;
        case atlas::proto::UNIVERSE_BULK_VELOCITY:
            universe_snapshot.bulk_velocity = unpack_vector3_buffer(state.vector_buffer(), "Universe::Builder::with_binary/bulk_velocity");
            break;
        case atlas::proto::UNIVERSE_FIELD_FORCE:
            universe_snapshot.field_force = unpack_vector3_buffer(state.vector_buffer(), "Universe::Builder::with_binary/field_force");
            break;
        case atlas::proto::UNIVERSE_MAX_RELATIVE_SPEED:
            universe_snapshot.max_relative_speed = unpack_scalar_buffer<float>(state.scalar_buffer(), "Universe::Builder::with_binary/max_relative_speed");
            break;
        case atlas::proto::UNIVERSE_THERMAL_ENERGY:
            universe_snapshot.thermal_energy = unpack_scalar_buffer<float>(state.scalar_buffer(), "Universe::Builder::with_binary/thermal_energy");
            break;
        case atlas::proto::UNIVERSE_NUMBER_PARTICLE:
            universe_snapshot.number_particle = unpack_scalar_buffer<float>(state.scalar_buffer(), "Universe::Builder::with_binary/number_particle");
            break;
        case atlas::proto::UNIVERSE_COLLISION_COUNT:
            universe_snapshot.collision_count = unpack_scalar_buffer<int>(state.int_buffer(), "Universe::Builder::with_binary/collision_count");
            break;
        case atlas::proto::UNIVERSE_KNUDSEN_NUMBER:
            universe_snapshot.knudsen_number = unpack_scalar_buffer<float>(state.scalar_buffer(), "Universe::Builder::with_binary/knudsen_number");
            break;
        default:
            throw std::runtime_error(
                "Universe::Builder::with_binary: encountered an unknown universe state kind.");
        }
    }

    return universe_snapshot;
}

} // namespace atlas
