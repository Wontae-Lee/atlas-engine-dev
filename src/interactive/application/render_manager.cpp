#include "application/render_manager.h"

#include "rendering/layer/geometry_layer.h"
#include "rendering/layer/particle_layer.h"
#include "rendering/renderer.h"
#include "rendering/state/raw_state_provider.h"
#include "rendering/target/window_target.h"
#include "server/server.h"
#include "session/session.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>

namespace atlas::interactive {

RenderManager::RenderManager() = default;

RenderManager::~RenderManager() {
    close();
}

void
RenderManager::open(const std::uint64_t session_id) {
    if (_session_id == session_id) return;
    if (_session_id) throw std::logic_error("renderer is already open for another session");

    auto window = std::make_unique<WindowTarget>(1280, 720, "Atlas Interactive");
    auto renderer = std::make_unique<Renderer>(std::make_unique<RawStateProvider>());
    renderer->add_layer(std::make_unique<GeometryLayer>());
    renderer->add_layer(std::make_unique<ParticleLayer>(6.0f));
    renderer->initialize(*window);

    _window = std::move(window);
    _renderer = std::move(renderer);
    _session_id = session_id;
    _next_frame = std::chrono::steady_clock::now();
}

void
RenderManager::close() {
    if (_renderer) _renderer->shutdown();
    _renderer.reset();
    _window.reset();
    _session_id.reset();
}

void
RenderManager::update(const Server& server) {
    if (!_session_id) return;
    const auto now = std::chrono::steady_clock::now();
    if (now < _next_frame) return;
    _next_frame = now + std::chrono::milliseconds(16);

    _window->poll_events(_renderer->camera());
    if (_window->should_close()) {
        close();
        return;
    }
    _renderer->render(server.session(*_session_id).scene_view(), *_window);
}

std::optional<std::uint64_t>
RenderManager::session_id() const noexcept {
    return _session_id;
}

}
