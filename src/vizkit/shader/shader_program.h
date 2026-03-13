#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
#include <vizkit/macros/macros.h>

namespace atlas::vizkit {

/**
 * @brief Lightweight RAII wrapper around an OpenGL shader program object.
 *
 * `ShaderProgram` owns one linked OpenGL program composed of:
 * - one vertex shader compiled from source text
 * - one fragment shader compiled from source text
 *
 * The class provides the minimum operations needed by the VizKit render path:
 * - compile and link a program during construction
 * - bind the program for subsequent draw calls
 * - query uniform locations by name
 *
 * Ownership is lifetime-based:
 * - successful construction creates and links an OpenGL program
 * - destruction releases the program with `glDeleteProgram`
 *
 * @note Construction may throw `std::runtime_error` if either shader fails to
 * compile or if the final program fails to link.
 */
class ShaderProgram {
public:
    /**
     * @brief Compiles and links a shader program from GLSL source strings.
     *
     * @param vs Null-terminated vertex shader source code.
     * @param fs Null-terminated fragment shader source code.
     *
     * Construction flow:
     * 1. compile the vertex shader
     * 2. compile the fragment shader
     * 3. create a program object
     * 4. attach both shaders
     * 5. link the program
     * 6. delete the temporary shader objects after linking
     *
     * @throws std::runtime_error If compilation or linking fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    ShaderProgram(const char* vs, const char* fs);

    /**
     * @brief Destroys the owned OpenGL program if one exists.
     *
     * The destructor is intentionally minimal and assumes a valid OpenGL
     * context is still available when the object is destroyed.
     */
    ~ShaderProgram() {
        if (_prog) glDeleteProgram(_prog);
    }

    /**
     * @brief Makes this program the current OpenGL program.
     *
     * After calling this function, subsequent uniform updates and draw calls
     * operate against the program owned by this object until another program is
     * bound.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    use() const;

    /**
     * @brief Queries the location of a uniform variable in the linked program.
     *
     * @param name Null-terminated uniform name as written in GLSL.
     * @return OpenGL uniform location, or `-1` if the uniform is not active or
     * does not exist in the linked program.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE GLint
    uniform_loc(const char* name) const;

private:
    /** @brief Owned OpenGL program handle, or `0` when uninitialized. */
    GLuint _prog { 0 };

    /**
     * @brief Compiles one shader stage from source text.
     *
     * @param type Shader stage enum such as `GL_VERTEX_SHADER` or
     * `GL_FRAGMENT_SHADER`.
     * @param src Null-terminated GLSL source code for the requested stage.
     * @return OpenGL shader object handle for the compiled shader.
     *
     * @throws std::runtime_error If shader compilation fails.
     *
     * @note The returned shader object is an intermediate resource intended to
     * be attached to a program and then deleted after linking.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static GLuint
    compile(GLenum type, const char* src);
};

}

#include <vizkit/shader/shader_program.hpp>

#endif
