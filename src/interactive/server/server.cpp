/**
 * @file
 * @brief Implements command dispatch and session lifecycle management.
 */

#include "server/server.h"

#include "config/system_factory.h"
#include "session/session.h"

#include <stdexcept>
#include <type_traits>
#include <variant>

namespace atlas::interactive {

Server::Server() = default;
Server::~Server() = default;

Response
Server::handle(const Request& request) {
    Response response;
    response.request_id = request.request_id;
    try {
        // Creation and shutdown are server-scoped and therefore need no session id.
        if (request.command == Command::create) {
            if (!request.simulation) {
                throw std::invalid_argument("create requires a simulation configuration");
            }
            auto session = std::make_unique<Session>(*request.simulation, request.output);
            const std::uint64_t id = _next_session_id++;
            _sessions.emplace(id, std::move(session));
            response.success = true;
            response.message = "created";
            response.session_id = id;
            response.status = _sessions.at(id)->status();
            return response;
        }
        if (request.command == Command::validate) {
            if (!request.validation) {
                throw std::invalid_argument("validate requires a target configuration");
            }
            CoreFactory factory;
            std::visit([&factory](const auto& config) {
                using Config = std::decay_t<decltype(config)>;
                if constexpr (std::is_same_v<Config, GeneratorValidation>) {
                    const auto materials = factory.build_materials(config.materials);
                    static_cast<void>(factory.build(config.generator, materials));
                } else if constexpr (std::is_same_v<Config, EmitterValidation>) {
                    const auto materials = factory.build_materials(config.materials);
                    static_cast<void>(factory.build(config.emitter.source));
                    static_cast<void>(factory.build(config.emitter.generator, materials));
                } else {
                    static_cast<void>(factory.build(config));
                }
            }, request.validation->config);
            response.success = true;
            response.message = "valid";
            return response;
        }
        if (request.command == Command::shutdown) {
            _shutdown_requested = true;
            response.success = true;
            response.message = "shutdown requested";
            return response;
        }
        if (request.command == Command::render_open ||
            request.command == Command::render_close) {
            throw std::invalid_argument("render commands require InteractiveApplication");
        }
        if (!request.session_id) throw std::invalid_argument("command requires session_id");
        const std::uint64_t id = *request.session_id;
        if (request.command == Command::close) {
            if (_sessions.erase(id) == 0) throw std::out_of_range("unknown session_id");
            response.success = true;
            response.message = "closed";
            response.session_id = id;
            return response;
        }

        // All remaining commands operate on one already-owned session.
        Session& target = session(id);
        switch (request.command) {
        case Command::start:
            target.start();
            response.message = "started";
            break;
        case Command::pause:
            target.pause();
            response.message = "paused";
            break;
        case Command::step:
            target.step(request.step_count);
            response.message = "stepped";
            break;
        case Command::status:
            response.message = "status";
            break;
        case Command::save:
            target.save(request.path);
            response.message = "saved";
            break;
        case Command::restart:
            target.restart();
            response.message = "restarted";
            break;
        case Command::create:
        case Command::validate:
        case Command::close:
        case Command::render_open:
        case Command::render_close:
        case Command::shutdown:
            break;
        }
        response.success = true;
        response.session_id = id;
        response.status = target.status();
    } catch (const std::exception& error) {
        response.error = error.what();
    }
    return response;
}

void
Server::update() {
    for (auto& [id, value] : _sessions) {
        static_cast<void>(id);
        value->update();
    }
}

bool
Server::shutdown_requested() const noexcept {
    return _shutdown_requested;
}

bool
Server::has_running_sessions() const noexcept {
    for (const auto& [id, value] : _sessions) {
        static_cast<void>(id);
        if (value->running()) return true;
    }
    return false;
}

std::size_t
Server::session_count() const noexcept {
    return _sessions.size();
}

Session&
Server::session(const std::uint64_t session_id) {
    const auto found = _sessions.find(session_id);
    if (found == _sessions.end()) throw std::out_of_range("unknown session_id");
    return *found->second;
}

const Session&
Server::session(const std::uint64_t session_id) const {
    const auto found = _sessions.find(session_id);
    if (found == _sessions.end()) throw std::out_of_range("unknown session_id");
    return *found->second;
}

}
