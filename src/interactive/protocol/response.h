/**
 * @file
 * @brief Defines a transport-independent interactive response.
 */

#pragma once

#include "session/session_status.h"

#include <optional>
#include <cstdint>
#include <string>

namespace atlas::interactive {

/// Transport-independent result returned for one request.
struct Response {
    std::string request_id; ///< Correlation identifier copied from the request.
    bool success = false; ///< Whether the command completed successfully.
    std::string message; ///< Human-readable success description.
    std::string error; ///< Human-readable failure description.
    std::optional<std::uint64_t> session_id; ///< Created or affected session identifier.
    std::optional<SessionStatus> status; ///< Current status when relevant to the command.
};

}
