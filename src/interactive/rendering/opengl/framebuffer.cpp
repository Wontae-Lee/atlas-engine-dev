/**
 * @file
 * @brief Implements offscreen framebuffer allocation and binding.
 */

#include "rendering/opengl/framebuffer.h"

#include <GL/glew.h>

#include <stdexcept>

namespace atlas::interactive::opengl {

Framebuffer::~Framebuffer() {
    destroy();
}

void
Framebuffer::resize(const int width, const int height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("Framebuffer dimensions must be positive.");
    if (_width == width && _height == height) return;
    create();

    glBindTexture(GL_TEXTURE_2D, _color_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindRenderbuffer(GL_RENDERBUFFER, _depth_buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);

    bind();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _color_texture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, _depth_buffer);
    // Validate attachment compatibility before exposing the resized framebuffer.
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        unbind();
        throw std::runtime_error("OpenGL framebuffer is incomplete.");
    }
    unbind();
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    _width = width;
    _height = height;
}

void
Framebuffer::bind() const {
    glBindFramebuffer(GL_FRAMEBUFFER, _id);
}

void
Framebuffer::unbind() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

unsigned int
Framebuffer::id() const noexcept {
    return _id;
}

unsigned int
Framebuffer::color_texture() const noexcept {
    return _color_texture;
}

int
Framebuffer::width() const noexcept {
    return _width;
}

int
Framebuffer::height() const noexcept {
    return _height;
}

void
Framebuffer::create() {
    if (_id != 0) return;
    glGenFramebuffers(1, &_id);
    glGenTextures(1, &_color_texture);
    glGenRenderbuffers(1, &_depth_buffer);
}

void
Framebuffer::destroy() noexcept {
    if (_depth_buffer != 0) glDeleteRenderbuffers(1, &_depth_buffer);
    if (_color_texture != 0) glDeleteTextures(1, &_color_texture);
    if (_id != 0) glDeleteFramebuffers(1, &_id);
    _id = 0;
    _color_texture = 0;
    _depth_buffer = 0;
    _width = 0;
    _height = 0;
}

}
