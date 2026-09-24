#include "application/interactive_application.h"

#include <stdexcept>
#include <utility>

namespace atlas::interactive {

InteractiveApplication::InteractiveApplication(std::unique_ptr<RenderControl> rendering)
    : _rendering(std::move(rendering)) {}

InteractiveApplication::~InteractiveApplication() = default;

Response
InteractiveApplication::handle(const Request& request) {
    if (request.command == Command::render_open || request.command == Command::render_close) {
        Response response;
        response.request_id = request.request_id;
        try {
            if (!_rendering) throw std::runtime_error("Rendering is unavailable in this build.");
            if (!request.session_id) throw std::invalid_argument("render command requires session_id");
            const std::uint64_t id = *request.session_id;
            static_cast<void>(_server.session(id));
            if (request.command == Command::render_open) {
                _rendering->open(id);
                response.message = "render opened";
            } else {
                const auto active = _rendering->session_id();
                if (active && *active != id) {
                    throw std::invalid_argument("renderer is open for another session");
                }
                _rendering->close();
                response.message = "render closed";
            }
            response.success = true;
            response.session_id = id;
        } catch (const std::exception& error) {
            response.error = error.what();
        }
        return response;
    }

    if (_rendering) {
        if (request.command == Command::close && request.session_id &&
            _rendering->session_id() == request.session_id) {
            _rendering->close();
        } else if (request.command == Command::shutdown) {
            _rendering->close();
        }
    }
    return _server.handle(request);
}

void
InteractiveApplication::update() {
    _server.update();
    if (_rendering) _rendering->update(_server);
}

bool
InteractiveApplication::shutdown_requested() const noexcept {
    return _server.shutdown_requested();
}

bool
InteractiveApplication::has_running_sessions() const noexcept {
    return _server.has_running_sessions();
}

bool
InteractiveApplication::rendering_active() const noexcept {
    return _rendering && _rendering->session_id().has_value();
}

std::size_t
InteractiveApplication::session_count() const noexcept {
    return _server.session_count();
}

}
