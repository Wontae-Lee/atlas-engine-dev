/**
 * @file
 * @brief Implements OpenGL buffer ownership and data upload.
 */

#include "rendering/opengl/buffer.h"

#include <GL/glew.h>

#include <utility>

namespace atlas::interactive::opengl {

Buffer::Buffer(Buffer&& other) noexcept
    : _id(std::exchange(other._id, 0))
    , _size(std::exchange(other._size, 0))
    , _capacity(std::exchange(other._capacity, 0)) {}

Buffer::~Buffer() {
    destroy();
}

Buffer&
Buffer::operator=(Buffer&& other) noexcept {
    if (this == &other) return *this;
    destroy();
    _id = std::exchange(other._id, 0);
    _size = std::exchange(other._size, 0);
    _capacity = std::exchange(other._capacity, 0);
    return *this;
}

void
Buffer::bind() const {
    glBindBuffer(GL_ARRAY_BUFFER, _id);
}

void
Buffer::unbind() {
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
Buffer::allocate(const std::size_t bytes) {
    if (bytes == 0) {
        _size = 0;
        return;
    }
    create();
    bind();
    if (bytes > _capacity) {
        // Capacity only grows, preserving storage and CUDA registration in steady state.
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, GL_DYNAMIC_DRAW);
        _capacity = bytes;
    }
    _size = bytes;
}

void
Buffer::upload(const void* source, const std::size_t bytes) {
    allocate(bytes);
    if (bytes == 0) return;
    bind();
    glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(bytes), source);
}

unsigned int
Buffer::id() const noexcept {
    return _id;
}

std::size_t
Buffer::size() const noexcept {
    return _size;
}

std::size_t
Buffer::capacity() const noexcept {
    return _capacity;
}

void
Buffer::create() {
    if (_id == 0) glGenBuffers(1, &_id);
}

void
Buffer::destroy() noexcept {
    if (_id != 0) glDeleteBuffers(1, &_id);
    _id = 0;
    _size = 0;
    _capacity = 0;
}

}
