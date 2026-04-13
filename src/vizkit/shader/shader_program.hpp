#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
namespace atlas::vizkit {

inline ShaderProgram::~ShaderProgram() {
    // Destroy the linked OpenGL program owned by this wrapper.
    //
    // This releases the GPU-side program object created in the constructor.
    //
    // Assumption:
    // - _prog is either a valid OpenGL program handle or zero-like state acceptable
    //   to glDeleteProgram().
    glDeleteProgram(_prog);
}

inline ShaderProgram::ShaderProgram(const char* vs, const char* fs) {
    // Build a complete shader program from vertex and fragment shader sources.
    //
    // High-level workflow:
    // 1. compile vertex shader source
    // 2. compile fragment shader source
    // 3. create program object
    // 4. attach compiled shaders
    // 5. link the program
    // 6. delete temporary shader objects
    // 7. validate link result
    {
        // Compile the vertex shader source into an OpenGL shader object.
        const GLuint v = compile(GL_VERTEX_SHADER, vs);

        // Compile the fragment shader source into an OpenGL shader object.
        const GLuint f = compile(GL_FRAGMENT_SHADER, fs);

        // Create the program object that will own the linked shader pipeline.
        _prog = glCreateProgram();

        // Attach the compiled vertex shader to the program.
        glAttachShader(_prog, v);

        // Attach the compiled fragment shader to the program.
        glAttachShader(_prog, f);

        // Link all attached shader stages into one executable GPU program.
        glLinkProgram(_prog);

        // Shader objects are no longer needed after linking.
        //
        // The linked program keeps its own internal copy of the compiled stages,
        // so the temporary shader handles can be deleted now.
        glDeleteShader(v);
        glDeleteShader(f);

        // Query whether program linking succeeded.
        GLint ok = GL_FALSE;
        glGetProgramiv(_prog, GL_LINK_STATUS, &ok);

        if (!ok) {
            // Retrieve the program link error log for diagnostics.
            char log[1024];
            glGetProgramInfoLog(_prog, 1024, nullptr, log);

            // Surface the link failure as a C++ exception with the OpenGL error text.
            throw std::runtime_error(std::string("Program link error: ") + log);
        }
    }
}

inline void
ShaderProgram::use() const {
    // Bind this shader program as the current active OpenGL program.
    //
    // After this call, subsequent draw calls and uniform uploads operate on _prog
    // until another program is bound.
    glUseProgram(_prog);
}

inline GLint
ShaderProgram::uniform_loc(const char* name) const {
    // Query the location of a named uniform variable in the linked program.
    //
    // Return value:
    // - non-negative location if the uniform exists and is active
    // - negative value if the uniform is missing or optimized out
    return glGetUniformLocation(_prog, name);
}

inline GLuint
ShaderProgram::compile(GLenum type, const char* src) {
    // Compile one shader stage from GLSL source text.
    //
    // Inputs:
    // - type : OpenGL shader stage enum, typically:
    //   * GL_VERTEX_SHADER
    //   * GL_FRAGMENT_SHADER
    // - src  : null-terminated GLSL source string
    //
    // Return value:
    // - compiled shader object handle on success
    //
    // Error behavior:
    // - throws std::runtime_error if compilation fails

    // Create the shader object for the requested stage.
    const GLuint s = glCreateShader(type);

    // Provide the GLSL source code to the shader object.
    glShaderSource(s, 1, &src, nullptr);

    // Compile the shader source into GPU-consumable form.
    glCompileShader(s);

    // Query whether compilation succeeded.
    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);

    if (!ok) {
        // Retrieve the shader compilation log for diagnostics.
        char log[1024];
        glGetShaderInfoLog(s, 1024, nullptr, log);

        // Produce a short human-readable stage tag for the error message.
        const std::string kind = (type == GL_VERTEX_SHADER) ? "VS" : "FS";

        // Surface the compilation failure as a C++ exception.
        throw std::runtime_error(kind + " compile error: " + log);
    }

    // Return the successfully compiled shader handle.
    return s;
}

}

#endif