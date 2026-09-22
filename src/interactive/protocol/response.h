#pragma once

#include "session/session_status.h"

#include <optional>
#include <string>

namespace atlas::interactive {

struct Response {
    bool success = false;
    std::string message;
    std::optional<SessionStatus> status;
};

}
