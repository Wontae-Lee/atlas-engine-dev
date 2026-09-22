#pragma once

#include "rendering/opengl/buffer.h"

#include <cstddef>
#include <optional>

namespace atlas::interactive {

struct RenderState {
    std::size_t particle_count = 0;
    opengl::Buffer position;
    opengl::Buffer velocity;
    opengl::Buffer species;
    std::optional<opengl::Buffer> temperature;
    std::optional<opengl::Buffer> translational_energy;
    std::optional<opengl::Buffer> rotational_energy;
    std::optional<opengl::Buffer> vibrational_energy;
};

}
