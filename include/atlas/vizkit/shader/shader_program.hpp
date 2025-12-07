#ifndef INCLUDE_VIZKIT_SHADER_SHADER_PROGRAM_HPP
#define INCLUDE_VIZKIT_SHADER_SHADER_PROGRAM_HPP

namespace atlas::vizkit {
ATLAS_HOST ATLAS_FORCE_INLINE
ShaderProgram::ShaderProgram(const char* vs, const char* fs) {
    {
        GLuint v = compile(GL_VERTEX_SHADER, vs);
        GLuint f = compile(GL_FRAGMENT_SHADER, fs);
        _prog    = glCreateProgram();
        glAttachShader(_prog, v);
        glAttachShader(_prog, f);
        glLinkProgram(_prog);
        glDeleteShader(v);
        glDeleteShader(f);
        GLint ok = GL_FALSE;
        glGetProgramiv(_prog, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetProgramInfoLog(_prog, 1024, nullptr, log);
            throw std::runtime_error(std::string("Program link error: ") + log);
        }
    }
}

ATLAS_HOST ATLAS_FORCE_INLINE void
ShaderProgram::use() const {
    glUseProgram(_prog);
}

ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE GLint
ShaderProgram::uniform_loc(const char* name) const {
    return glGetUniformLocation(_prog, name);
}

ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE GLuint
ShaderProgram::compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, 1024, nullptr, log);
        std::string kind = (type == GL_VERTEX_SHADER) ? "VS" : "FS";
        throw std::runtime_error(kind + " compile error: " + log);
    }
    return s;
}

}
#endif