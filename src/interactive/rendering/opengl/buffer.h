#pragma once

#include <cstddef>

namespace atlas::interactive::opengl {

class Buffer final {
public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    ~Buffer();

    Buffer& operator=(const Buffer&) = delete;
    Buffer& operator=(Buffer&& other) noexcept;

    void bind() const;
    static void unbind();
    void allocate(std::size_t bytes);
    void upload(const void* source, std::size_t bytes);

    unsigned int id() const noexcept;
    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;

private:
    void create();
    void destroy() noexcept;

    unsigned int _id = 0;
    std::size_t _size = 0;
    std::size_t _capacity = 0;
};

}
