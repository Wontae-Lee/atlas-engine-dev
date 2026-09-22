#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace atlas::interactive::opengl {
class Buffer;
}

namespace atlas::interactive {

class StateBridge final {
public:
    StateBridge();
    StateBridge(const StateBridge&) = delete;
    StateBridge(StateBridge&&) = delete;
    ~StateBridge();

    StateBridge& operator=(const StateBridge&) = delete;
    StateBridge& operator=(StateBridge&&) = delete;

    template <typename T>
    void upload(const DeviceBuffer<T>& source,
                const std::size_t count,
                opengl::Buffer& target) {
        if (count > source.size()) throw std::out_of_range("StateBridge upload exceeds the source buffer.");
        upload_raw(atlas::raw_pointer_cast(source.data()), count * sizeof(T), target);
    }

    void release(opengl::Buffer& target);

private:
    struct Impl;

    void upload_raw(const void* source, std::size_t bytes, opengl::Buffer& target);

    std::unique_ptr<Impl> _impl;
};

}
