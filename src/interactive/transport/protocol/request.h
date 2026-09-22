#pragma once

#include "transport/protocol/command.h"

#include <cstddef>

namespace atlas::interactive::transport {

struct Request {
    Command command = Command::step;
    std::size_t step_count = 1;
};

}
