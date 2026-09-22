#pragma once

#include "protocol/command.h"

#include <cstddef>
#include <filesystem>

namespace atlas::interactive {

struct Request {
    Command command = Command::step;
    std::size_t step_count = 1;
    std::filesystem::path path;
};

}
