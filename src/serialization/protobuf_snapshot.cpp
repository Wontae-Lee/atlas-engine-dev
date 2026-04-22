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
#include <type_traits>

namespace atlas::serialization {
namespace {

constexpr std::uint32_t atlas_snapshot_version = 1u;

template <typename T>
atlas::proto::ScalarType
protobuf_scalar_type() {
    if constexpr (std::is_same_v<T, float>) {
        return atlas::proto::SCALAR_FLOAT32;
    } else if constexpr (std::is_same_v<T, double>) {
        return atlas::proto::SCALAR_FLOAT64;
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "protobuf restart snapshots only support float and double.");
    }
}

template <typename T>
void
validate_scalar_type(const atlas::proto::ScalarType scalar_type,
                     const char* context) {
    if (scalar_type != protobuf_scalar_type<T>()) {
        throw std::runtime_error(std::string(context)
                                 + ": scalar type in snapshot does not match the requested template type.");
    }
}

inline void
set_optional_double(atlas::proto::OptionalDouble* target,
                    const std::optional<double>& value) {
    target->set_has_value(value.has_value());
    if (value.has_value()) {
        target->set_value(*value);
    }
}

template <typename T>
void
set_optional_double(atlas::proto::OptionalDouble* target,
                    const std::optional<T>& value) {
    target->set_has_value(value.has_value());
    if (value.has_value()) {
        target->set_value(static_cast<double>(*value));
    }
}

inline void
set_optional_int32(atlas::proto::OptionalInt32* target,
                   const std::optional<int>& value) {
    target->set_has_value(value.has_value());
    if (value.has_value()) {
        target->set_value(*value);
    }
}

template <typename T>
std::optional<T>
read_optional_double(const atlas::proto::OptionalDouble& value) {
    if (!value.has_value()) {
        return std::nullopt;
    }

    return static_cast<T>(value.value());
}

inline std::optional<int>
read_optional_int32(const atlas::proto::OptionalInt32& value) {
    if (!value.has_value()) {
        return std::nullopt;
    }

    return value.value();
}

template <typename T>
void
set_vector3(atlas::proto::Vector3* target,
            const Vector3<T>& value) {
    target->set_x(static_cast<double>(value.x));
    target->set_y(static_cast<double>(value.y));
    target->set_z(static_cast<double>(value.z));
}

template <typename T>
Vector3<T>
read_vector3(const atlas::proto::Vector3& value) {
    return Vector3<T>(
        static_cast<T>(value.x()),
        static_cast<T>(value.y()),
        static_cast<T>(value.z()));
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
unpack_scalar_buffer(const atlas::proto::RawBuffer& buffer,
                     const char* context) {
    if (buffer.element_count() == 0) {
        return {};
    }

    const auto expected_size = static_cast<std::size_t>(buffer.element_count()) * sizeof(T);
    if (buffer.data().size() != static_cast<int>(expected_size)) {
        throw std::runtime_error(std::string(context) + ": raw buffer byte size does not match the encoded element count.");
    }

    HostBuffer<T> values(static_cast<std::size_t>(buffer.element_count()));
    std::memcpy(values.data(), buffer.data().data(), expected_size);
    return values;
}

template <typename T>
std::string
pack_vector3_buffer(const HostBuffer<Vector3<T>>& values) {
    if (values.empty()) {
        return {};
    }

    std::string bytes;
    bytes.resize(values.size() * sizeof(Vector3<T>));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

template <typename T>
HostBuffer<Vector3<T>>
unpack_vector3_buffer(const atlas::proto::RawBuffer& buffer,
                      const char* context) {
    if (buffer.element_count() == 0) {
        return {};
    }

    const auto expected_size = static_cast<std::size_t>(buffer.element_count()) * sizeof(Vector3<T>);
    if (buffer.data().size() != static_cast<int>(expected_size)) {
        throw std::runtime_error(std::string(context) + ": vector buffer byte size does not match the encoded element count.");
    }

    HostBuffer<Vector3<T>> values(static_cast<std::size_t>(buffer.element_count()));
    std::memcpy(values.data(), buffer.data().data(), expected_size);
    return values;
}

template <typename T>
void
set_material_property(atlas::proto::MaterialProperty* target,
                      const MatrialProperties<T>& source) {
    target->set_type(static_cast<int>(source.type));
    target->set_mass(static_cast<double>(source.mass));
    target->set_molecular_mass(static_cast<double>(source.molecular_mass));
    set_optional_double(target->mutable_translational_energy(), source.translational_energy);
    set_optional_double(target->mutable_rotational_energy(), source.rotational_energy);
    set_optional_double(target->mutable_vibrational_energy(), source.vibrational_energy);
    set_optional_int32(target->mutable_species_id(), source.species_id);
    set_optional_double(target->mutable_collision_diameter(), source.collision_diameter);
    set_optional_double(target->mutable_viscosity_index(), source.viscosity_index);
    set_optional_double(target->mutable_scattering_parameter(), source.scattering_parameter);
    set_optional_double(target->mutable_rest_density(), source.rest_density);
    set_optional_double(target->mutable_pressure_coefficient(), source.pressure_coefficient);
    set_optional_double(target->mutable_dynamic_viscosity(), source.dynamic_viscosity);
    set_optional_double(target->mutable_smoothing_length(), source.smoothing_length);
    set_optional_double(target->mutable_electronic_energy(), source.electronic_energy);
    set_optional_int32(target->mutable_charge(), source.charge);
}

template <typename T>
MatrialProperties<T>
read_material_property(const atlas::proto::MaterialProperty& source) {
    MatrialProperties<T> result {};
    result.type = static_cast<atlas::system::MaterialType::Value>(source.type());
    result.mass = static_cast<T>(source.mass());
    result.molecular_mass = static_cast<T>(source.molecular_mass());
    result.translational_energy = read_optional_double<T>(source.translational_energy());
    result.rotational_energy = read_optional_double<T>(source.rotational_energy());
    result.vibrational_energy = read_optional_double<T>(source.vibrational_energy());
    result.species_id = read_optional_int32(source.species_id());
    result.collision_diameter = read_optional_double<T>(source.collision_diameter());
    result.viscosity_index = read_optional_double<T>(source.viscosity_index());
    result.scattering_parameter = read_optional_double<T>(source.scattering_parameter());
    result.rest_density = read_optional_double<T>(source.rest_density());
    result.pressure_coefficient = read_optional_double<T>(source.pressure_coefficient());
    result.dynamic_viscosity = read_optional_double<T>(source.dynamic_viscosity());
    result.smoothing_length = read_optional_double<T>(source.smoothing_length());
    result.electronic_energy = read_optional_double<T>(source.electronic_energy());
    result.charge = read_optional_int32(source.charge());
    return result;
}

template <typename T>
atlas::proto::GeneratorType
to_proto_generator_type(const fluid::GenerateType type) {
    switch (type) {
        case fluid::GenerateType::maxwell_sigma:
            return atlas::proto::GENERATOR_MAXWELL_SIGMA;
        case fluid::GenerateType::maxwell_boltzmann:
            return atlas::proto::GENERATOR_MAXWELL_BOLTZMANN;
        case fluid::GenerateType::uniform:
            return atlas::proto::GENERATOR_UNIFORM;
        default:
            throw std::runtime_error("Unsupported generator operator type.");
    }
}

template <typename T>
fluid::GenerateOperator<T>
read_generator(const atlas::proto::GeneratorOperator& source) {
    switch (source.type()) {
        case atlas::proto::GENERATOR_UNIFORM:
            return fluid::GenerateOperator<T>(fluid::UniformGenerateOperator<T>(source.seed()));
        case atlas::proto::GENERATOR_MAXWELL_SIGMA:
            return fluid::GenerateOperator<T>(fluid::MaxwellSigmaGenerateOperator<T>(source.seed()));
        case atlas::proto::GENERATOR_MAXWELL_BOLTZMANN:
            return fluid::GenerateOperator<T>(
                fluid::MaxwellBoltzmannGenerateOperator<T>(
                    source.seed(),
                    read_vector3<T>(source.bulk_velocity())));
        default:
            throw std::runtime_error("Fluid snapshot contains an unknown generator type.");
    }
}

template <typename T>
void
set_generator(atlas::proto::GeneratorOperator* target,
              const fluid::GenerateOperator<T>& source) {
    target->set_type(to_proto_generator_type<T>(source.type));

    switch (source.type) {
        case fluid::GenerateType::uniform:
            target->set_seed(source.uniform.seed);
            break;
        case fluid::GenerateType::maxwell_sigma:
            target->set_seed(source.maxwell_sigma.seed);
            break;
        case fluid::GenerateType::maxwell_boltzmann:
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

template <typename T>
void
set_vector_raw_buffer(atlas::proto::RawBuffer* target,
                      const HostBuffer<Vector3<T>>& values) {
    target->set_element_count(values.size());
    target->set_data(pack_vector3_buffer(values));
}

template <typename T>
void
append_universe_scalar_state(atlas::proto::UniverseSnapshot* snapshot,
                             const atlas::proto::UniverseStateKind kind,
                             const HostBuffer<T>& values) {
    auto* state = snapshot->add_states();
    state->set_kind(kind);
    set_raw_buffer(state->mutable_scalar_buffer(), values);
}

template <typename T>
void
append_universe_vector_state(atlas::proto::UniverseSnapshot* snapshot,
                             const atlas::proto::UniverseStateKind kind,
                             const HostBuffer<Vector3<T>>& values) {
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

template <typename T>
void
save_fluid_binary(const atlas::fluid::Fluid<T>& fluid, std::string_view path) {
    atlas::proto::FluidSnapshot snapshot;
    snapshot.set_version(atlas_snapshot_version);
    snapshot.set_scalar_type(protobuf_scalar_type<T>());
    snapshot.set_buffer_size(fluid.buffer_size());
    snapshot.set_particle_count(fluid.particle_count());
    snapshot.set_statistical_weight(static_cast<double>(fluid.statistical_weight()));

    {
        HostBuffer<MatrialProperties<T>> properties(
            fluid.particle_properties().begin(),
            fluid.particle_properties().end());

        for (const auto& property : properties) {
            set_material_property(snapshot.add_properties(), property);
        }
    }

    {
        HostBuffer<fluid::GenerateOperator<T>> generators(
            fluid.generators().begin(),
            fluid.generators().end());

        for (const auto& generator : generators) {
            set_generator(snapshot.add_generators(), generator);
        }
    }

    std::size_t known_state_count = 0;

    if (const auto* state = fluid.template state<atlas::fluid::FluidPositionState<T>>(); state != nullptr) {
        const HostBuffer<Vector3<T>> values(state->data().begin(), state->data().end());
        set_vector_raw_buffer(snapshot.mutable_positions(), values);
        ++known_state_count;
    }

    if (const auto* state = fluid.template state<atlas::fluid::FluidVelocityState<T>>(); state != nullptr) {
        const HostBuffer<Vector3<T>> values(state->data().begin(), state->data().end());
        set_vector_raw_buffer(snapshot.mutable_velocities(), values);
        ++known_state_count;
    }

    if (const auto* state = fluid.template state<atlas::fluid::FluidSpeciesState<T>>(); state != nullptr) {
        const HostBuffer<std::size_t> values(state->data().begin(), state->data().end());
        set_raw_buffer(snapshot.mutable_species(), values);
        ++known_state_count;
    }

    if (const auto* state = fluid.template state<atlas::fluid::FluidActiveState<T>>(); state != nullptr) {
        const HostBuffer<int> values(state->data().begin(), state->data().end());
        set_raw_buffer(snapshot.mutable_active(), values);
        ++known_state_count;
    }

    if (const auto* state = fluid.template state<atlas::fluid::FluidTemperatureState<T>>(); state != nullptr) {
        const HostBuffer<T> values(state->data().begin(), state->data().end());
        set_raw_buffer(snapshot.mutable_temperature(), values);
        snapshot.set_includes_temperature_state(true);
        ++known_state_count;
    }

    if (fluid.states().size() != known_state_count) {
        throw std::runtime_error("Fluid::save: encountered an unsupported fluid state type during protobuf serialization.");
    }

    write_message(snapshot, path, "Fluid::save");
}

template <typename T>
FluidBinarySnapshot<T>
load_fluid_binary(std::string_view path) {
    const auto snapshot = read_message<atlas::proto::FluidSnapshot>(path, "Fluid::Builder::with_binary");

    if (snapshot.version() != atlas_snapshot_version) {
        throw std::runtime_error("Fluid::Builder::with_binary: unsupported fluid snapshot version.");
    }

    validate_scalar_type<T>(snapshot.scalar_type(), "Fluid::Builder::with_binary");

    FluidBinarySnapshot<T> result;
    result.buffer_size = static_cast<std::size_t>(snapshot.buffer_size());
    result.particle_count = static_cast<std::size_t>(snapshot.particle_count());
    result.statistical_weight = static_cast<T>(snapshot.statistical_weight());

    result.properties.reserve(snapshot.properties_size());
    for (const auto& property : snapshot.properties()) {
        result.properties.push_back(read_material_property<T>(property));
    }

    result.generators.reserve(snapshot.generators_size());
    for (const auto& generator : snapshot.generators()) {
        result.generators.push_back(read_generator<T>(generator));
    }

    if (snapshot.positions().element_count() > 0) {
        result.positions = unpack_vector3_buffer<T>(snapshot.positions(), "Fluid::Builder::with_binary/positions");
    }

    if (snapshot.velocities().element_count() > 0) {
        result.velocities = unpack_vector3_buffer<T>(snapshot.velocities(), "Fluid::Builder::with_binary/velocities");
    }

    if (snapshot.species().element_count() > 0) {
        result.species = unpack_scalar_buffer<std::size_t>(snapshot.species(), "Fluid::Builder::with_binary/species");
    }

    if (snapshot.active().element_count() > 0) {
        result.active = unpack_scalar_buffer<int>(snapshot.active(), "Fluid::Builder::with_binary/active");
    }

    if (snapshot.includes_temperature_state()) {
        result.temperature = unpack_scalar_buffer<T>(snapshot.temperature(), "Fluid::Builder::with_binary/temperature");
    }

    if (result.particle_count > result.buffer_size) {
        throw std::runtime_error("Fluid::Builder::with_binary: particle_count exceeds buffer_size in snapshot.");
    }

    const auto validate_state_size = [&](const auto& state, const char* name) {
        if (state.has_value() && state->size() != result.buffer_size) {
            throw std::runtime_error(std::string("Fluid::Builder::with_binary: ")
                                     + name + " state size does not match buffer_size.");
        }
    };

    validate_state_size(result.positions, "position");
    validate_state_size(result.velocities, "velocity");
    validate_state_size(result.species, "species");
    validate_state_size(result.active, "active");
    validate_state_size(result.temperature, "temperature");

    return result;
}

template <typename T>
void
save_universe_binary(const atlas::universe::Universe<T>& universe, std::string_view path) {
    atlas::proto::UniverseSnapshot snapshot;
    snapshot.set_version(atlas_snapshot_version);
    snapshot.set_scalar_type(protobuf_scalar_type<T>());
    set_vector3(snapshot.mutable_lower_corner(), universe.lower_corner());
    set_vector3(snapshot.mutable_upper_corner(), universe.upper_corner());
    snapshot.set_cell_size(static_cast<double>(universe.cell_size()));

    std::size_t known_state_count = 0;

    if (const auto* state = universe.template state<atlas::universe::UniverseTemperatureState<T>>(); state != nullptr) {
        append_universe_scalar_state(&snapshot,
                                     atlas::proto::UNIVERSE_TEMPERATURE,
                                     HostBuffer<T>(state->data().begin(), state->data().end()));
        ++known_state_count;
    }

    if (const auto* state = universe.template state<atlas::universe::UniverseBulkVelocityState<T>>(); state != nullptr) {
        append_universe_vector_state(&snapshot,
                                     atlas::proto::UNIVERSE_BULK_VELOCITY,
                                     HostBuffer<Vector3<T>>(state->data().begin(), state->data().end()));
        ++known_state_count;
    }

    if (const auto* state = universe.template state<atlas::universe::UniverseFieldForceState<T>>(); state != nullptr) {
        append_universe_vector_state(&snapshot,
                                     atlas::proto::UNIVERSE_FIELD_FORCE,
                                     HostBuffer<Vector3<T>>(state->data().begin(), state->data().end()));
        ++known_state_count;
    }

    if (const auto* state = universe.template state<atlas::universe::UniverseMaxRelativeSpeedState<T>>(); state != nullptr) {
        append_universe_scalar_state(&snapshot,
                                     atlas::proto::UNIVERSE_MAX_RELATIVE_SPEED,
                                     HostBuffer<T>(state->data().begin(), state->data().end()));
        ++known_state_count;
    }

    if (const auto* state = universe.template state<atlas::universe::UniverseThermalEnergyState<T>>(); state != nullptr) {
        append_universe_scalar_state(&snapshot,
                                     atlas::proto::UNIVERSE_THERMAL_ENERGY,
                                     HostBuffer<T>(state->data().begin(), state->data().end()));
        ++known_state_count;
    }

    if (const auto* state = universe.template state<atlas::universe::UniverseNumberParticleState<T>>(); state != nullptr) {
        append_universe_scalar_state(&snapshot,
                                     atlas::proto::UNIVERSE_NUMBER_PARTICLE,
                                     HostBuffer<T>(state->data().begin(), state->data().end()));
        ++known_state_count;
    }

    if (const auto* state = universe.template state<atlas::universe::UniverseCollisionCountState<int>>(); state != nullptr) {
        append_universe_int_state(&snapshot,
                                  atlas::proto::UNIVERSE_COLLISION_COUNT,
                                  HostBuffer<int>(state->data().begin(), state->data().end()));
        ++known_state_count;
    }

    if (const auto* state = universe.template state<atlas::universe::UniverseKnudsenNumberState<T>>(); state != nullptr) {
        append_universe_scalar_state(&snapshot,
                                     atlas::proto::UNIVERSE_KNUDSEN_NUMBER,
                                     HostBuffer<T>(state->data().begin(), state->data().end()));
        ++known_state_count;
    }

    if (universe.states().size() != known_state_count) {
        throw std::runtime_error("Universe::save: encountered an unsupported universe state type during protobuf serialization.");
    }

    write_message(snapshot, path, "Universe::save");
}

template <typename T>
UniverseBinarySnapshot<T>
load_universe_binary(std::string_view path) {
    const auto snapshot = read_message<atlas::proto::UniverseSnapshot>(path, "Universe::Builder::with_binary");

    if (snapshot.version() != atlas_snapshot_version) {
        throw std::runtime_error("Universe::Builder::with_binary: unsupported universe snapshot version.");
    }

    validate_scalar_type<T>(snapshot.scalar_type(), "Universe::Builder::with_binary");

    UniverseBinarySnapshot<T> result;
    result.lower_corner = read_vector3<T>(snapshot.lower_corner());
    result.upper_corner = read_vector3<T>(snapshot.upper_corner());
    result.cell_size = static_cast<T>(snapshot.cell_size());

    for (const auto& state : snapshot.states()) {
        switch (state.kind()) {
            case atlas::proto::UNIVERSE_TEMPERATURE:
                result.temperature = unpack_scalar_buffer<T>(state.scalar_buffer(), "Universe::Builder::with_binary/temperature");
                break;
            case atlas::proto::UNIVERSE_BULK_VELOCITY:
                result.bulk_velocity = unpack_vector3_buffer<T>(state.vector_buffer(), "Universe::Builder::with_binary/bulk_velocity");
                break;
            case atlas::proto::UNIVERSE_FIELD_FORCE:
                result.field_force = unpack_vector3_buffer<T>(state.vector_buffer(), "Universe::Builder::with_binary/field_force");
                break;
            case atlas::proto::UNIVERSE_MAX_RELATIVE_SPEED:
                result.max_relative_speed = unpack_scalar_buffer<T>(state.scalar_buffer(), "Universe::Builder::with_binary/max_relative_speed");
                break;
            case atlas::proto::UNIVERSE_THERMAL_ENERGY:
                result.thermal_energy = unpack_scalar_buffer<T>(state.scalar_buffer(), "Universe::Builder::with_binary/thermal_energy");
                break;
            case atlas::proto::UNIVERSE_NUMBER_PARTICLE:
                result.number_particle = unpack_scalar_buffer<T>(state.scalar_buffer(), "Universe::Builder::with_binary/number_particle");
                break;
            case atlas::proto::UNIVERSE_COLLISION_COUNT:
                result.collision_count = unpack_scalar_buffer<int>(state.int_buffer(), "Universe::Builder::with_binary/collision_count");
                break;
            case atlas::proto::UNIVERSE_KNUDSEN_NUMBER:
                result.knudsen_number = unpack_scalar_buffer<T>(state.scalar_buffer(), "Universe::Builder::with_binary/knudsen_number");
                break;
            default:
                throw std::runtime_error("Universe::Builder::with_binary: encountered an unknown universe state kind.");
        }
    }

    return result;
}

template void save_fluid_binary<float>(const atlas::fluid::Fluid<float>&, std::string_view);
template void save_fluid_binary<double>(const atlas::fluid::Fluid<double>&, std::string_view);
template FluidBinarySnapshot<float> load_fluid_binary<float>(std::string_view);
template FluidBinarySnapshot<double> load_fluid_binary<double>(std::string_view);

template void save_universe_binary<float>(const atlas::universe::Universe<float>&, std::string_view);
template void save_universe_binary<double>(const atlas::universe::Universe<double>&, std::string_view);
template UniverseBinarySnapshot<float> load_universe_binary<float>(std::string_view);
template UniverseBinarySnapshot<double> load_universe_binary<double>(std::string_view);

} // namespace atlas::serialization
