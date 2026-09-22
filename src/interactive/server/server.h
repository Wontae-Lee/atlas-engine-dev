#pragma once

#include "protocol/request.h"
#include "protocol/response.h"

namespace atlas::interactive {

class Session;

class Server final {
public:
    explicit Server(Session& session) noexcept;

    Response handle(const Request& request);
    bool shutdown_requested() const noexcept;

private:
    Session* _session = nullptr;
};

}
