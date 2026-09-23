/**
 * @file
 * @brief Declares an owning OpenGL buffer wrapper.
 */

#pragma once

#include <cstddef>

namespace atlas::interactive::opengl {

/// Move-only owner of an OpenGL array-buffer object.
class Buffer final {
public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    ~Buffer();

    Buffer& operator=(const Buffer&) = delete;
    Buffer& operator=(Buffer&& other) noexcept;

    /// Binds this object to GL_ARRAY_BUFFER.
    void bind() const;
    /// Clears the current GL_ARRAY_BUFFER binding.
    static void unbind();
    /// Ensures capacity for bytes and resets the logical size.
    void allocate(std::size_t bytes);
    /// Uploads bytes and records them as the logical buffer size.
    void upload(const void* source, std::size_t bytes);

    /// Returns the OpenGL object name, or zero before creation.
    unsigned int id() const noexcept;
    /// Returns the number of bytes containing current data.
    std::size_t size() const noexcept;
    /// Returns the number of bytes allocated by OpenGL.
    std::size_t capacity() const noexcept;

private:
    void create();
    void destroy() noexcept;

    unsigned int _id = 0; ///< OpenGL object name.
    std::size_t _size = 0; ///< Logical uploaded size in bytes.
    std::size_t _capacity = 0; ///< Allocated storage in bytes.
};

}
