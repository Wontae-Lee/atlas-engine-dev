/**
 * @file
 * @brief Implements shader compilation, linking, and ownership.
 */

#include "rendering/opengl/shader.h"

#include <GL/glew.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace atlas::interactive::opengl {

Shader::Shader(const std::string_view vertex_source, const std::string_view fragment_source) {
    const unsigned int vertex = compile(GL_VERTEX_SHADER, vertex_source);
    unsigned int fragment = 0;
    try {
        fragment = compile(GL_FRAGMENT_SHADER, fragment_source);
        _id = glCreateProgram();
        glAttachShader(_id, vertex);
        glAttachShader(_id, fragment);
        glLinkProgram(_id);

        int linked = GL_FALSE;
        glGetProgramiv(_id, GL_LINK_STATUS, &linked);
        if (linked != GL_TRUE) {
            int length = 0;
            glGetProgramiv(_id, GL_INFO_LOG_LENGTH, &length);
            std::string log(static_cast<std::size_t>(length), '\0');
            glGetProgramInfoLog(_id, length, nullptr, log.data());
            throw std::runtime_error("OpenGL program link failed: " + log);
        }
    } catch (...) {
        // Construction must not leak the first shader when later compilation or linking fails.
        if (_id != 0) glDeleteProgram(_id);
        glDeleteShader(vertex);
        if (fragment != 0) glDeleteShader(fragment);
        _id = 0;
        throw;
    }
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

Shader::Shader(Shader&& other) noexcept
    : _id(std::exchange(other._id, 0)) {}

Shader::~Shader() {
    destroy();
}

Shader&
Shader::operator=(Shader&& other) noexcept {
    if (this == &other) return *this;
    destroy();
    _id = std::exchange(other._id, 0);
    return *this;
}

void
Shader::use() const {
    glUseProgram(_id);
}

int
Shader::uniform_location(const char* name) const {
    return glGetUniformLocation(_id, name);
}

unsigned int
Shader::id() const noexcept {
    return _id;
}

unsigned int
Shader::compile(const unsigned int type, const std::string_view source) {
    const unsigned int shader = glCreateShader(type);
    const char* data = source.data();
    const int length = static_cast<int>(source.size());
    glShaderSource(shader, 1, &data, &length);
    glCompileShader(shader);

    int compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE) return shader;

    int log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    std::string log(static_cast<std::size_t>(log_length), '\0');
    glGetShaderInfoLog(shader, log_length, nullptr, log.data());
    glDeleteShader(shader);
    throw std::runtime_error("OpenGL shader compilation failed: " + log);
}

void
Shader::destroy() noexcept {
    if (_id != 0) glDeleteProgram(_id);
    _id = 0;
}

}
