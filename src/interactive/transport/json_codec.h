/**
 * @file
 * @brief Declares JSON conversion for interactive configuration and protocol values.
 */

#pragma once

#include "config/simulation_config.h"
#include "protocol/request.h"
#include "protocol/response.h"

#include <string>
#include <string_view>

namespace atlas::interactive {

/// Converts protocol and configuration values to and from JSON text.
class JsonCodec final {
public:
    /// Decodes one complete simulation configuration document.
    static SimulationConfig decode_simulation(std::string_view text);
    /// Decodes one protocol request document.
    static Request decode_request(std::string_view text, std::string* request_id = nullptr);
    /// Encodes one protocol response document.
    static std::string encode_response(const Response& response);
};

}
