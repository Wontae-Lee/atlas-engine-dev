#pragma once

#include "transport/protocol/request.h"
#include "transport/protocol/response.h"

namespace atlas::interactive {
class Session;
}

namespace atlas::interactive::transport {

class Server final {
public:
    explicit Server(Session& session) noexcept;

    Response handle(const Request& request);
    bool shutdown_requested() const noexcept;

private:
    Session* _session = nullptr;
    bool _shutdown_requested = false;
};

}
