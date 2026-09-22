#pragma once

#include "transport/transport.h"

#include <iosfwd>
#include <mutex>

namespace atlas::interactive {

class JsonLinesTransport final : public Transport {
public:
    JsonLinesTransport(std::istream& input, std::ostream& output) noexcept;

    bool receive(Request& request, Response& error) override;
    void send(const Response& response) override;

private:
    std::istream* _input;
    std::ostream* _output;
    std::mutex _output_mutex;
};

}
