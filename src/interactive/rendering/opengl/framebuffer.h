/**
 * @file
 * @brief Declares an owning offscreen OpenGL framebuffer.
 */

#pragma once

namespace atlas::interactive::opengl {

/// Owns an offscreen framebuffer with color and depth attachments.
class Framebuffer final {
public:
    Framebuffer() = default;
    Framebuffer(const Framebuffer&) = delete;
    Framebuffer(Framebuffer&&) = delete;
    ~Framebuffer();

    Framebuffer& operator=(const Framebuffer&) = delete;
    Framebuffer& operator=(Framebuffer&&) = delete;

    /// Allocates or resizes all attachments to the supplied dimensions.
    void resize(int width, int height);
    /// Binds this framebuffer as the current draw/read target.
    void bind() const;
    /// Restores the default framebuffer binding.
    static void unbind();

    /// Returns the framebuffer object name.
    unsigned int id() const noexcept;
    /// Returns the color texture object name.
    unsigned int color_texture() const noexcept;
    /// Returns the attachment width in pixels.
    int width() const noexcept;
    /// Returns the attachment height in pixels.
    int height() const noexcept;

private:
    void create();
    void destroy() noexcept;

    unsigned int _id = 0; ///< Framebuffer object name.
    unsigned int _color_texture = 0; ///< RGBA8 color attachment.
    unsigned int _depth_buffer = 0; ///< Depth-stencil renderbuffer attachment.
    int _width = 0; ///< Allocated width in pixels.
    int _height = 0; ///< Allocated height in pixels.
};

}
