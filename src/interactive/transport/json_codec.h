#pragma once

#include "config/simulation_config.h"
#include "protocol/request.h"
#include "protocol/response.h"

#include <string>
#include <string_view>

namespace atlas::interactive {

class JsonCodec final {
public:
    static SimulationConfig decode_simulation(std::string_view text);
    static Request decode_request(std::string_view text);
    static std::string encode_response(const Response& response);
};

}
