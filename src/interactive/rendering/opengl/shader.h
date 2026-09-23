/**
 * @file
 * @brief Declares an owning OpenGL shader-program wrapper.
 */

#pragma once

#include <string_view>

namespace atlas::interactive::opengl {

/// Move-only owner of a linked OpenGL shader program.
class Shader final {
public:
    Shader() = default;
    /// Compiles and links a vertex/fragment shader pair.
    Shader(std::string_view vertex_source, std::string_view fragment_source);
    Shader(const Shader&) = delete;
    Shader(Shader&& other) noexcept;
    ~Shader();

    Shader& operator=(const Shader&) = delete;
    Shader& operator=(Shader&& other) noexcept;

    /// Makes this program current.
    void use() const;
    /// Returns a uniform location, or -1 when it is inactive.
    int uniform_location(const char* name) const;
    /// Returns the OpenGL program object name.
    unsigned int id() const noexcept;

private:
    static unsigned int compile(unsigned int type, std::string_view source);
    void destroy() noexcept;

    unsigned int _id = 0; ///< Linked OpenGL program object name.
};

}
