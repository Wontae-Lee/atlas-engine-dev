#pragma once

#include <string_view>

namespace atlas::interactive::opengl {

class Shader final {
public:
    Shader() = default;
    Shader(std::string_view vertex_source, std::string_view fragment_source);
    Shader(const Shader&) = delete;
    Shader(Shader&& other) noexcept;
    ~Shader();

    Shader& operator=(const Shader&) = delete;
    Shader& operator=(Shader&& other) noexcept;

    void use() const;
    int uniform_location(const char* name) const;
    unsigned int id() const noexcept;

private:
    static unsigned int compile(unsigned int type, std::string_view source);
    void destroy() noexcept;

    unsigned int _id = 0;
};

}
