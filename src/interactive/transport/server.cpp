#include "transport/server.h"

#include "session/session.h"

#include <exception>

namespace atlas::interactive::transport {

Server::Server(Session& session) noexcept
    : _session(&session) {}

Response
Server::handle(const Request& request) {
    try {
        switch (request.command) {
            case Command::initialize:
                return { false, "initialize requires a native System payload." };
            case Command::start:
                _session->start();
                return { true, "started" };
            case Command::pause:
                _session->pause();
                return { true, "paused" };
            case Command::step:
                _session->step(request.step_count);
                return { true, "stepped" };
            case Command::close:
                _session->close();
                return { true, "closed" };
            case Command::shutdown:
                _shutdown_requested = true;
                return { true, "shutdown requested" };
        }
    } catch (const std::exception& error) {
        return { false, error.what() };
    }
    return { false, "unknown command" };
}

bool
Server::shutdown_requested() const noexcept {
    return _shutdown_requested;
}

}
