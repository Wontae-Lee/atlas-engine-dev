#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

/**
 * @file shader_program.h
 * @brief Declares a minimal OpenGL shader program wrapper used by Vizkit layers.
 *
 * @details
 * This header defines @ref atlas::vizkit::ShaderProgram, a lightweight RAII
 * wrapper around an OpenGL shader program object.
 *
 * ## Purpose
 * The class provides a minimal abstraction for:
 * - compiling vertex and fragment shaders from source strings,
 * - linking them into a program,
 * - activating the program for rendering,
 * - querying uniform locations.
 *
 * It is intentionally simple and designed for internal use within Vizkit layers
 * such as:
 * - geometry layers,
 * - particle layers,
 * - custom visualization components.
 *
 * ## Lifetime management
 * The class follows RAII principles:
 * - shader program creation occurs in the constructor,
 * - shader program deletion occurs in the destructor.
 *
 * This ensures that OpenGL resources are automatically released when the object
 * goes out of scope.
 *
 * ## Typical usage
 * @code
 * ShaderProgram program(vertex_shader_src, fragment_shader_src);
 * program.use();
 *
 * GLint mvp_loc = program.uniform_loc("u_mvp");
 * @endcode
 *
 * ## Error handling
 * Shader compilation and linking errors are expected to be handled internally
 * in the implementation (`shader_program.hpp`), typically via:
 * - OpenGL error checks,
 * - log output,
 * - or assertions depending on build configuration.
 *
 * ---
 */

#include <vizkit/macros/macros.h>

namespace atlas::vizkit {

/**
 * @brief RAII wrapper for an OpenGL shader program.
 *
 * @details
 * @ref ShaderProgram encapsulates:
 * - creation of a program object,
 * - compilation of vertex and fragment shaders,
 * - linking of the final program,
 * - activation of the program for rendering.
 *
 * ## Design goals
 * - minimal overhead,
 * - no dynamic polymorphism,
 * - simple interface tailored for internal engine usage,
 * - compatibility with both debug and release builds.
 *
 * ## Responsibilities
 * This class is responsible for:
 * - compiling shader source strings into OpenGL shader objects,
 * - linking those objects into a program,
 * - providing access to uniform locations,
 * - binding the program for rendering.
 *
 * ## Non-responsibilities
 * This class does **not** manage:
 * - uniform value uploads,
 * - attribute binding beyond default usage,
 * - shader source generation,
 * - advanced pipeline features (UBOs, SSBOs, etc.).
 *
 * ---
 */
class ShaderProgram {
public:
    /**
     * @brief Construct and link a shader program from source strings.
     *
     * @details
     * This constructor:
     * 1. compiles the provided vertex shader source,
     * 2. compiles the provided fragment shader source,
     * 3. links both into a single OpenGL program,
     * 4. stores the resulting program handle.
     *
     * @param vs Null-terminated vertex shader GLSL source string.
     * @param fs Null-terminated fragment shader GLSL source string.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    ShaderProgram(const char* vs, const char* fs);

    /**
     * @brief Destroy the shader program and release OpenGL resources.
     *
     * @details
     * Deletes the underlying OpenGL program object if it exists.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    ~ShaderProgram();

    /**
     * @brief Bind this shader program for subsequent draw calls.
     *
     * @details
     * Internally calls `glUseProgram` with the stored program handle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    use() const;

    /**
     * @brief Query the location of a uniform variable.
     *
     * @details
     * This function retrieves the location of a uniform variable from the
     * linked shader program.
     *
     * @param name Null-terminated string containing the uniform name.
     * @return Location of the uniform, or -1 if not found.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE GLint
    uniform_loc(const char* name) const;

    /**
     * @brief Upload a 4x4 float matrix uniform if it exists in the shader.
     *
     * @param name Uniform name.
     * @param value Pointer to 16 contiguous float values.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_mat4(const char* name, const float* value) const;

    /**
     * @brief Upload an RGBA float vector uniform if it exists in the shader.
     *
     * @param name Uniform name.
     * @param x First component.
     * @param y Second component.
     * @param z Third component.
     * @param w Fourth component.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_vec4(const char* name, float x, float y, float z, float w) const;

    /**
     * @brief Upload a scalar float uniform if it exists in the shader.
     *
     * @param name Uniform name.
     * @param value Uniform value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_float(const char* name, float value) const;

private:
    /**
     * @brief OpenGL program handle.
     */
    GLuint _prog { 0 };

    /**
     * @brief Compile a shader from source.
     *
     * @details
     * This helper function:
     * - creates a shader object of the given type,
     * - attaches the provided source string,
     * - compiles the shader,
     * - returns the compiled shader handle.
     *
     * Error handling (e.g., compilation failure logging) is expected to be
     * implemented in the corresponding `.hpp`.
     *
     * @param type OpenGL shader type (e.g., GL_VERTEX_SHADER, GL_FRAGMENT_SHADER).
     * @param src Null-terminated GLSL source string.
     * @return Compiled shader object handle.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static GLuint
    compile(GLenum type, const char* src);
};

} // namespace atlas::vizkit

#include <vizkit/shader/shader_program.hpp>

#endif
