#pragma once

#include "application/render_control.h"
#include "protocol/request.h"
#include "protocol/response.h"
#include "server/server.h"

#include <cstddef>
#include <memory>

namespace atlas::interactive {

class InteractiveApplication final {
public:
    explicit InteractiveApplication(std::unique_ptr<RenderControl> rendering = {});
    ~InteractiveApplication();

    Response handle(const Request& request);
    void update();

    bool shutdown_requested() const noexcept;
    bool has_running_sessions() const noexcept;
    bool rendering_active() const noexcept;
    std::size_t session_count() const noexcept;

private:
    Server _server;
    std::unique_ptr<RenderControl> _rendering;
};

}
