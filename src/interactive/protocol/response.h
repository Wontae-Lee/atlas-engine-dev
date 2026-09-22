#pragma once

#include "session/session_status.h"

#include <optional>
#include <cstdint>
#include <string>

namespace atlas::interactive {

struct Response {
    std::string request_id;
    bool success = false;
    std::string message;
    std::string error;
    std::optional<std::uint64_t> session_id;
    std::optional<SessionStatus> status;
};

}
