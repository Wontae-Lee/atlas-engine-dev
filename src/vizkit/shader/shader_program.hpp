#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
namespace atlas::vizkit {
ShaderProgram::ShaderProgram(const char* vs, const char* fs) {
    {
        // Compile each shader stage independently first.
        //
        // OpenGL separates "shader objects" from the final "program object":
        // - shader objects hold compiled source for one stage
        // - the program object links several stages into one executable pipeline
        //
        // Here we build:
        // - one vertex shader   : transforms vertices into clip space
        // - one fragment shader : computes output pixel color
        const GLuint v = compile(GL_VERTEX_SHADER, vs);
        const GLuint f = compile(GL_FRAGMENT_SHADER, fs);

        // Create the program object that will own the linked executable.
        _prog = glCreateProgram();

        // Attach the compiled stages to the program prior to linking.
        // At this moment OpenGL still treats them as separate shader modules.
        glAttachShader(_prog, v);
        glAttachShader(_prog, f);

        // Link performs interface matching between stages and produces the
        // final GPU program. Typical checks include:
        // - vertex shader outputs vs fragment shader inputs
        // - uniform/interface block compatibility
        // - implementation-specific compilation into executable form
        glLinkProgram(_prog);

        // Once linking has been requested, the standalone shader objects are no
        // longer needed by this wrapper. The linked program keeps the necessary
        // internal representation, so the temporary shader handles can be
        // deleted without invalidating the final program.
        glDeleteShader(v);
        glDeleteShader(f);

        // Query the link result explicitly. OpenGL reports failures through
        // status flags and info logs rather than C++ exceptions.
        GLint ok = GL_FALSE;
        glGetProgramiv(_prog, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[1024];

            // Fetch the diagnostic log produced by the driver/linker. This
            // usually contains stage-interface mismatches or other GLSL errors
            // that could not be detected until the full program was assembled.
            glGetProgramInfoLog(_prog, 1024, nullptr, log);
            throw std::runtime_error(std::string("Program link error: ") + log);
        }
    }
}

void
ShaderProgram::use() const {
    // Bind this program into the current OpenGL context.
    //
    // After this call, subsequent uniform updates and draw calls operate
    // against _prog until another program is bound with glUseProgram.
    glUseProgram(_prog);
}

GLint
ShaderProgram::uniform_loc(const char* name) const {
    // Uniform lookup is name-based after program linking.
    // The returned integer is the handle used by glUniform* calls.
    //
    // A return value of -1 means the uniform is absent, optimized out, or not
    // active in the final linked program.
    return glGetUniformLocation(_prog, name);
}

GLuint
ShaderProgram::compile(GLenum type, const char* src) {
    // Create one shader-stage object. The type selects which GLSL stage this
    // source belongs to, for example vertex or fragment.
    const GLuint s = glCreateShader(type);

    // Associate source text with the shader object.
    // The second parameter is the number of source strings supplied; here it is
    // 1 because the whole shader is passed as one contiguous C string.
    glShaderSource(s, 1, &src, nullptr);

    // Ask the driver to compile GLSL source into stage-specific GPU code.
    glCompileShader(s);

    // As with program linking, shader compilation success must be queried via
    // the OpenGL status API.
    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];

        // The shader info log typically includes:
        // - syntax errors
        // - type mismatches
        // - use of unsupported GLSL features
        // - stage-specific semantic issues
        glGetShaderInfoLog(s, 1024, nullptr, log);
        const std::string kind = (type == GL_VERTEX_SHADER) ? "VS" : "FS";
        throw std::runtime_error(kind + " compile error: " + log);
    }

    // Return the compiled shader handle to the caller so it can be attached to
    // a program and linked with the other stages.
    return s;
}

}

#endif
