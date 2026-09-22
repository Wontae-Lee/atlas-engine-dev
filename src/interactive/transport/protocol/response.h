#pragma once

#include <string>

namespace atlas::interactive::transport {

struct Response {
    bool success = false;
    std::string message;
};

}
