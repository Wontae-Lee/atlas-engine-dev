#pragma once

#include "application/render_control.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

namespace atlas::interactive {

class Renderer;
class WindowTarget;

class RenderManager final : public RenderControl {
public:
    RenderManager();
    ~RenderManager() override;

    void open(std::uint64_t session_id) override;
    void close() override;
    void update(const Server& server) override;
    std::optional<std::uint64_t> session_id() const noexcept override;

private:
    std::optional<std::uint64_t> _session_id;
    std::unique_ptr<WindowTarget> _window;
    std::unique_ptr<Renderer> _renderer;
    std::chrono::steady_clock::time_point _next_frame {};
};

}
