/**
 * @file
 * @brief Implements strict JSON decoding and response encoding.
 */

#include "transport/json_codec.h"

#include <nlohmann/json.hpp>

#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace atlas::interactive {

namespace {

using Json = nlohmann::json;
using Config = SimulationConfig;

/// Maps a protocol spelling to an enum and rejects unknown values.
template <typename Enum>
Enum
enum_value(const std::string& value,
           const std::initializer_list<std::pair<const char*, Enum>>& choices,
           const char* field) {
    for (const auto& [name, result] : choices) {
        if (value == name) return result;
    }
    throw std::invalid_argument(std::string("Invalid ") + field + ": " + value);
}

/// Decodes a strict three-component floating-point array.
Config::Vec3
vec3(const Json& value) {
    if (!value.is_array() || value.size() != 3) {
        throw std::invalid_argument("Expected a three-component vector.");
    }
    return { value.at(0).get<float>(), value.at(1).get<float>(), value.at(2).get<float>() };
}

/// Decodes a strict scalar-first quaternion array.
Config::Quat
quat(const Json& value) {
    if (!value.is_array() || value.size() != 4) {
        throw std::invalid_argument("Expected a four-component quaternion [w,x,y,z].");
    }
    return { value.at(0).get<float>(), value.at(1).get<float>(),
             value.at(2).get<float>(), value.at(3).get<float>() };
}

/// Overwrites target only when an optional JSON member is present.
template <typename T>
void
read(const Json& object, const char* name, T& target) {
    if (object.contains(name)) target = object.at(name).get<T>();
}

/// Reads an optional vector member with strict component validation.
void
read_vec3(const Json& object, const char* name, Config::Vec3& target) {
    if (object.contains(name)) target = vec3(object.at(name));
}

/// Decodes an optional scalar array while preserving absence.
template <typename T>
std::optional<std::vector<T>>
optional_array(const Json& object, const char* name) {
    if (!object.contains(name)) return std::nullopt;
    return object.at(name).get<std::vector<T>>();
}

/// Decodes an optional array of strict three-component vectors.
std::optional<std::vector<Config::Vec3>>
optional_vec3_array(const Json& object, const char* name) {
    if (!object.contains(name)) return std::nullopt;
    std::vector<Config::Vec3> result;
    for (const auto& value : object.at(name)) result.push_back(vec3(value));
    return result;
}

/// Decodes one material configuration object.
Config::Material
material(const Json& value) {
    Config::Material result;
    result.kind = enum_value<Config::MaterialKind>(
        value.value("type", "molecule"),
        { { "molecule", Config::MaterialKind::molecule },
          { "atom", Config::MaterialKind::atom },
          { "ion", Config::MaterialKind::ion },
          { "neutron", Config::MaterialKind::neutron },
          { "solid", Config::MaterialKind::solid } },
        "material type");
    read(value, "mass", result.mass);
    read(value, "translational_energy", result.translational_energy);
    read(value, "rotational_energy", result.rotational_energy);
    read(value, "vibrational_energy", result.vibrational_energy);
    read(value, "reference_diameter", result.reference_diameter);
    read(value, "reference_temperature", result.reference_temperature);
    read(value, "viscosity_index", result.viscosity_index);
    read(value, "scattering_parameter", result.scattering_parameter);
    return result;
}

/// Decodes one geometry configuration object.
Config::Geometry
geometry(const Json& value) {
    Config::Geometry result;
    result.kind = enum_value<Config::GeometryKind>(
        value.at("type").get<std::string>(),
        { { "box", Config::GeometryKind::box },
          { "circle", Config::GeometryKind::circle },
          { "cylinder", Config::GeometryKind::cylinder },
          { "plane", Config::GeometryKind::plane },
          { "sphere", Config::GeometryKind::sphere },
          { "square", Config::GeometryKind::square },
          { "triangle", Config::GeometryKind::triangle },
          { "triangle_mesh", Config::GeometryKind::triangle_mesh },
          { "polygonal_prism", Config::GeometryKind::polygonal_prism } },
        "geometry type");
    read_vec3(value, "lower_corner", result.lower_corner);
    read_vec3(value, "upper_corner", result.upper_corner);
    read_vec3(value, "center", result.center);
    read_vec3(value, "normal", result.normal);
    read_vec3(value, "a", result.a);
    read_vec3(value, "b", result.b);
    read_vec3(value, "c", result.c);
    read(value, "radius", result.radius);
    read(value, "height", result.height);
    read(value, "offset", result.offset);
    read(value, "side_length", result.side_length);
    read(value, "side_count", result.side_count);
    read(value, "open", result.open);
    if (value.contains("triangles")) {
        for (const auto& triangle : value.at("triangles")) {
            if (!triangle.is_array() || triangle.size() != 3) {
                throw std::invalid_argument("Triangle mesh faces require three vertices.");
            }
            result.triangles.push_back(
                { vec3(triangle.at(0)), vec3(triangle.at(1)), vec3(triangle.at(2)) });
        }
    }
    return result;
}

/// Decodes one geometry unit and its optional kinematics.
Config::Unit
unit(const Json& value) {
    Config::Unit result;
    result.geometry = geometry(value.at("geometry"));
    read_vec3(value, "translation", result.translation);
    if (value.contains("orientation")) result.orientation = quat(value.at("orientation"));
    if (value.contains("velocity")) result.velocity = vec3(value.at("velocity"));
    if (value.contains("acceleration")) result.acceleration = vec3(value.at("acceleration"));
    if (value.contains("angular_velocity")) {
        result.angular_velocity = vec3(value.at("angular_velocity"));
    }
    if (value.contains("angular_acceleration")) {
        result.angular_acceleration = vec3(value.at("angular_acceleration"));
    }
    return result;
}

/// Decodes initial Fluid storage and state arrays.
Config::Fluid
fluid(const Json& value) {
    Config::Fluid result;
    result.buffer_size = value.at("buffer_size").get<std::size_t>();
    read(value, "particle_count", result.particle_count);
    read(value, "statistical_weight", result.statistical_weight);
    if (value.contains("materials")) {
        for (const auto& entry : value.at("materials")) result.materials.push_back(material(entry));
    }
    if (value.contains("position")) {
        for (const auto& entry : value.at("position")) result.position.push_back(vec3(entry));
    }
    if (value.contains("velocity")) {
        for (const auto& entry : value.at("velocity")) result.velocity.push_back(vec3(entry));
    }
    if (value.contains("species")) result.species = value.at("species").get<std::vector<std::size_t>>();
    result.temperature = optional_array<float>(value, "temperature");
    result.translational_energy = optional_array<float>(value, "translational_energy");
    result.rotational_energy = optional_array<float>(value, "rotational_energy");
    result.vibrational_energy = optional_array<float>(value, "vibrational_energy");
    return result;
}

/// Decodes the Universe domain and optional cell states.
Config::Universe
universe(const Json& value) {
    Config::Universe result;
    read_vec3(value, "lower_corner", result.lower_corner);
    read_vec3(value, "upper_corner", result.upper_corner);
    if (value.contains("geometry")) result.geometry = geometry(value.at("geometry"));
    result.cell_size = value.at("cell_size").get<float>();
    result.temperature = optional_array<float>(value, "temperature");
    result.bulk_velocity = optional_vec3_array(value, "bulk_velocity");
    result.field_force = optional_vec3_array(value, "field_force");
    result.gravity = optional_vec3_array(value, "gravity");
    result.thermal_energy = optional_array<float>(value, "thermal_energy");
    result.knudsen_number = optional_array<float>(value, "knudsen_number");
    return result;
}

/// Decodes one DSMC solver configuration.
Config::Solver
solver(const Json& value) {
    Config::Solver result;
    result.kernel = enum_value<Config::DsmcKernelKind>(
        value.value("kernel", "hard_sphere"),
        { { "hard_sphere", Config::DsmcKernelKind::hard_sphere },
          { "variable_hard_sphere", Config::DsmcKernelKind::variable_hard_sphere },
          { "variable_soft_sphere", Config::DsmcKernelKind::variable_soft_sphere } },
        "DSMC kernel");
    read(value, "majorant_sample_pairs", result.majorant_sample_pairs);
    read(value, "majorant_exhaustive_limit", result.majorant_exhaustive_limit);
    return result;
}

/// Decodes one particle-source configuration.
Config::Source
source(const Json& value) {
    Config::Source result;
    result.kind = enum_value<Config::SourceKind>(
        value.value("type", "volume"),
        { { "surface", Config::SourceKind::surface },
          { "volume", Config::SourceKind::volume } },
        "source type");
    result.unit = unit(value.at("unit"));
    read(value, "tolerance", result.tolerance);
    read(value, "spacing", result.spacing);
    return result;
}

/// Decodes one emitted-particle generator configuration.
Config::Generator
generator(const Json& value) {
    Config::Generator result;
    result.kind = enum_value<Config::GeneratorKind>(
        value.value("type", "uniform"),
        { { "uniform", Config::GeneratorKind::uniform },
          { "jittering", Config::GeneratorKind::jittering },
          { "maxwell_sigma", Config::GeneratorKind::maxwell_sigma },
          { "maxwell_boltzmann", Config::GeneratorKind::maxwell_boltzmann } },
        "generator type");
    read(value, "species_ratios", result.species_ratios);
    read(value, "species_numbers", result.species_numbers);
    read(value, "species_mass", result.species_mass);
    read(value, "temperature", result.temperature);
    read(value, "min_value", result.min_value);
    read(value, "max_value", result.max_value);
    read(value, "base_value", result.base_value);
    read(value, "jitter_radius", result.jitter_radius);
    read(value, "sigma", result.sigma);
    read_vec3(value, "bulk_velocity", result.bulk_velocity);
    read(value, "seed", result.seed);
    return result;
}

/// Decodes one isothermal collider configuration.
Config::Collider
collider(const Json& value) {
    Config::Collider result;
    result.unit = unit(value.at("unit"));
    read(value, "momentum_accommodation_coefficient",
         result.momentum_accommodation_coefficient);
    read(value, "restitution", result.restitution);
    result.diffuse_sampling = enum_value<Config::DiffuseSamplingKind>(
        value.value("diffuse_sampling", "uniform"),
        { { "cosine_weighted", Config::DiffuseSamplingKind::cosine_weighted },
          { "uniform", Config::DiffuseSamplingKind::uniform } },
        "diffuse sampling");
    return result;
}

/// Decodes one particle-sink configuration.
Config::Sink
sink(const Json& value) {
    Config::Sink result;
    result.kind = enum_value<Config::SinkKind>(
        value.value("type", "volume"),
        { { "surface", Config::SinkKind::surface },
          { "volume", Config::SinkKind::volume },
          { "tracing", Config::SinkKind::tracing } },
        "sink type");
    result.unit = unit(value.at("unit"));
    read(value, "tolerance", result.tolerance);
    return result;
}

/// Decodes optional Knudsen codec parameters.
Config::Codec
codec(const Json& value) {
    Config::Codec result;
    read(value, "representative_characteristic_length",
         result.representative_characteristic_length);
    read(value, "representative_collision_cross_sectional_area",
         result.representative_collision_cross_sectional_area);
    read(value, "representative_statistical_weight",
         result.representative_statistical_weight);
    read(value, "representative_cell_volume", result.representative_cell_volume);
    return result;
}

/// Decodes a complete simulation description from a JSON object.
SimulationConfig
simulation(const Json& value) {
    SimulationConfig result;
    result.dt = value.at("dt").get<float>();
    result.fluid = fluid(value.at("fluid"));
    result.universe = universe(value.at("universe"));
    if (value.contains("solvers")) {
        for (const auto& entry : value.at("solvers")) result.solvers.push_back(solver(entry));
    }
    if (value.contains("emitters")) {
        for (const auto& entry : value.at("emitters")) {
            result.emitters.push_back({ source(entry.at("source")),
                                        generator(entry.at("generator")) });
        }
    }
    if (value.contains("colliders")) {
        for (const auto& entry : value.at("colliders")) result.colliders.push_back(collider(entry));
    }
    if (value.contains("sinks")) {
        for (const auto& entry : value.at("sinks")) result.sinks.push_back(sink(entry));
    }
    if (value.contains("codec")) result.codec = codec(value.at("codec"));
    return result;
}

/// Decodes application-owned output policy.
OutputConfig
output(const Json& value) {
    OutputConfig result;
    read(value, "csv_enabled", result.csv_enabled);
    if (value.contains("output_directory")) {
        result.output_directory = value.at("output_directory").get<std::string>();
    }
    read(value, "csv_filename", result.csv_filename);
    return result;
}

/// Maps a protocol command spelling to its enum value.
Command
command(const std::string& value) {
    return enum_value<Command>(
        value,
        { { "create", Command::create }, { "start", Command::start },
          { "pause", Command::pause }, { "step", Command::step },
          { "status", Command::status }, { "save", Command::save },
          { "restart", Command::restart }, { "close", Command::close },
          { "shutdown", Command::shutdown } },
        "command");
}

/// Returns the stable wire spelling for a session state.
const char*
state_name(const SessionState state) {
    switch (state) {
    case SessionState::ready: return "ready";
    case SessionState::running: return "running";
    case SessionState::paused: return "paused";
    }
    return "ready";
}

}

SimulationConfig
JsonCodec::decode_simulation(const std::string_view text) {
    return simulation(Json::parse(text));
}

Request
JsonCodec::decode_request(const std::string_view text) {
    const Json value = Json::parse(text);
    Request result;
    result.request_id = value.value("request_id", "");
    result.command = command(value.at("command").get<std::string>());
    if (value.contains("session_id")) result.session_id = value.at("session_id").get<std::uint64_t>();
    // Accept the flat form for local tools while preserving the protocol payload form.
    const Json& payload = value.contains("payload") ? value.at("payload") : value;
    if (payload.contains("step_count")) result.step_count = payload.at("step_count").get<std::size_t>();
    if (payload.contains("path")) result.path = payload.at("path").get<std::string>();
    if (payload.contains("config")) result.simulation = simulation(payload.at("config"));
    if (payload.contains("output")) result.output = output(payload.at("output"));
    return result;
}

std::string
JsonCodec::encode_response(const Response& response) {
    Json value = {
        { "request_id", response.request_id },
        { "success", response.success }
    };
    if (!response.message.empty()) value["message"] = response.message;
    if (!response.error.empty()) value["error"] = response.error;
    if (response.session_id) value["session_id"] = *response.session_id;
    if (response.status) {
        value["status"] = {
            { "state", state_name(response.status->state) },
            { "step", response.status->step },
            { "simulation_time", response.status->simulation_time },
            { "particle_count", response.status->particle_count },
            { "source_count", response.status->source_count },
            { "sink_count", response.status->sink_count }
        };
    }
    return value.dump();
}

}
