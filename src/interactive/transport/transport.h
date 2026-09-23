/**
 * @file
 * @brief Declares the transport-independent request and response channel.
 */

#pragma once

#include "protocol/request.h"
#include "protocol/response.h"

namespace atlas::interactive {

/// Abstract command channel independent of framing and client type.
class Transport {
public:
    virtual ~Transport() = default;
    /// Receives and decodes one request, returning false at end of input.
    virtual bool receive(Request& request, Response& error) = 0;
    /// Encodes and sends one response.
    virtual void send(const Response& response) = 0;
};

}
