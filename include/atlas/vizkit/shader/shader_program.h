#ifndef INCLUDE_VIZKIT_SHADER_SHADER_PROGRAM_H
#define INCLUDE_VIZKIT_SHADER_SHADER_PROGRAM_H
#ifdef ATLAS_ENABLE_VIZKIT
#include <atlas/core/macros.h>

namespace atlas::vizkit {

class ShaderProgram {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE
    ShaderProgram(const char* vs, const char* fs);
    ~ShaderProgram() {
        if (_prog) glDeleteProgram(_prog);
    }
    ATLAS_HOST ATLAS_FORCE_INLINE void
    use() const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE GLint
    uniform_loc(const char* name) const;

private:
    GLuint _prog { 0 };

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static GLuint
    compile(GLenum type, const char* src);
};

}

#include <atlas/vizkit/shader/shader_program.hpp>

#endif
#endif