#pragma once

namespace atlas::interactive::opengl {

class Framebuffer final {
public:
    Framebuffer() = default;
    Framebuffer(const Framebuffer&) = delete;
    Framebuffer(Framebuffer&&) = delete;
    ~Framebuffer();

    Framebuffer& operator=(const Framebuffer&) = delete;
    Framebuffer& operator=(Framebuffer&&) = delete;

    void resize(int width, int height);
    void bind() const;
    static void unbind();

    unsigned int id() const noexcept;
    unsigned int color_texture() const noexcept;
    int width() const noexcept;
    int height() const noexcept;

private:
    void create();
    void destroy() noexcept;

    unsigned int _id = 0;
    unsigned int _color_texture = 0;
    unsigned int _depth_buffer = 0;
    int _width = 0;
    int _height = 0;
};

}
