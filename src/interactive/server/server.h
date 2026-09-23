/**
 * @file
 * @brief Declares multi-session command dispatch for interactive clients.
 */

#pragma once

#include "protocol/request.h"
#include "protocol/response.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace atlas::interactive {

class Session;

/// Dispatches protocol commands and owns all active sessions.
class Server final {
public:
    Server();
    ~Server();

    /// Handles one request synchronously and returns its response.
    Response handle(const Request& request);
    /// Advances every session currently in the running state once.
    void update();
    /// Reports whether a shutdown command has been accepted.
    bool shutdown_requested() const noexcept;
    /// Reports whether at least one session is running.
    bool has_running_sessions() const noexcept;
    /// Returns the number of owned sessions.
    std::size_t session_count() const noexcept;
    /// Returns a mutable session or throws when the identifier is unknown.
    Session& session(std::uint64_t session_id);
    /// Returns a session or throws when the identifier is unknown.
    const Session& session(std::uint64_t session_id) const;

private:
    std::unordered_map<std::uint64_t, std::unique_ptr<Session>> _sessions; ///< Owned sessions by id.
    std::uint64_t _next_session_id = 1; ///< Monotonic identifier for the next session.
    bool _shutdown_requested = false; ///< Latched shutdown state.
};

}
