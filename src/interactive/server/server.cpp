#include "server/server.h"

#include "session/session.h"

namespace atlas::interactive {

Server::Server(Session& session) noexcept
    : _session(&session) {}

Response
Server::handle(const Request& request) {
    return _session->handle(request);
}

bool
Server::shutdown_requested() const noexcept {
    return _session->shutdown_requested();
}

}
