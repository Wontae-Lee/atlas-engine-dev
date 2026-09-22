#pragma once

#include "protocol/request.h"
#include "protocol/response.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace atlas::interactive {

class Session;

class Server final {
public:
    Server();
    ~Server();

    Response handle(const Request& request);
    void update();
    bool shutdown_requested() const noexcept;
    bool has_running_sessions() const noexcept;
    std::size_t session_count() const noexcept;
    Session& session(std::uint64_t session_id);
    const Session& session(std::uint64_t session_id) const;

private:
    std::unordered_map<std::uint64_t, std::unique_ptr<Session>> _sessions;
    std::uint64_t _next_session_id = 1;
    bool _shutdown_requested = false;
};

}
