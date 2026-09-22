#pragma once

#include "protocol/command.h"
#include "config/output_config.h"
#include "config/simulation_config.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace atlas::interactive {

struct Request {
    std::string request_id;
    std::optional<std::uint64_t> session_id;
    Command command = Command::step;
    std::size_t step_count = 1;
    std::filesystem::path path;
    std::optional<SimulationConfig> simulation;
    OutputConfig output;
};

}
