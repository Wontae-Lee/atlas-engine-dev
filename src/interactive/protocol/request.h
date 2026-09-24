/**
 * @file
 * @brief Defines a transport-independent interactive request.
 */

#pragma once

#include "protocol/command.h"
#include "config/output_config.h"
#include "config/simulation_config.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace atlas::interactive {

struct GeneratorValidation {
    SimulationConfig::Generator generator;
    std::vector<SimulationConfig::Material> materials;
};

struct EmitterValidation {
    SimulationConfig::Emitter emitter;
    std::vector<SimulationConfig::Material> materials;
};

using ValidationConfig = std::variant<SimulationConfig::Material,
                                      SimulationConfig::Geometry,
                                      SimulationConfig::Unit,
                                      SimulationConfig::Fluid,
                                      SimulationConfig::Universe,
                                      SimulationConfig::Solver,
                                      SimulationConfig::Source,
                                      GeneratorValidation,
                                      EmitterValidation,
                                      SimulationConfig::Collider,
                                      SimulationConfig::Sink,
                                      SimulationConfig::Codec,
                                      SimulationConfig>;

struct ValidationRequest {
    ValidationConfig config;
};

/// Transport-independent command request accepted by Server.
struct Request {
    std::string request_id; ///< Client-provided correlation identifier.
    std::optional<std::uint64_t> session_id; ///< Target session for session-scoped commands.
    Command command = Command::step; ///< Requested operation.
    std::size_t step_count = 1; ///< Number of steps for a step command.
    std::filesystem::path path; ///< Destination used by persistence commands.
    std::optional<SimulationConfig> simulation; ///< Configuration required by create.
    std::optional<ValidationRequest> validation;
    OutputConfig output; ///< Application output policy for a new session.
};

}
