#pragma once

#include <cstdint>
#include <optional>

namespace atlas::interactive {

class Server;

class RenderControl {
public:
    virtual ~RenderControl() = default;

    virtual void open(std::uint64_t session_id) = 0;
    virtual void close() = 0;
    virtual void update(const Server& server) = 0;
    virtual std::optional<std::uint64_t> session_id() const noexcept = 0;
};

}
