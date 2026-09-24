/**
 * @file
 * @brief Implements newline-delimited JSON request and response transport.
 */

#include "transport/json_lines_transport.h"

#include "transport/json_codec.h"

#include <exception>
#include <istream>
#include <ostream>
#include <string>

namespace atlas::interactive {

JsonLinesTransport::JsonLinesTransport(std::istream& input, std::ostream& output) noexcept
    : _input(&input)
    , _output(&output) {}

bool
JsonLinesTransport::receive(Request& request, Response& error) {
    std::string line;
    while (std::getline(*_input, line)) {
        if (line.empty()) continue;
        std::string request_id;
        try {
            request = JsonCodec::decode_request(line, &request_id);
            return true;
        } catch (const std::exception& exception) {
            // A malformed line produces an error response without terminating the channel.
            error = {};
            error.request_id = request_id;
            error.error = exception.what();
            send(error);
        }
    }
    return false;
}

void
JsonLinesTransport::send(const Response& response) {
    // Hold the lock through flush so concurrent responses cannot share one JSON line.
    const std::lock_guard lock(_output_mutex);
    *_output << JsonCodec::encode_response(response) << '\n';
    _output->flush();
}

}
