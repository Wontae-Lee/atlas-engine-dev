#pragma once

#include "protocol/request.h"
#include "protocol/response.h"

namespace atlas::interactive {

class Transport {
public:
    virtual ~Transport() = default;
    virtual bool receive(Request& request, Response& error) = 0;
    virtual void send(const Response& response) = 0;
};

}
