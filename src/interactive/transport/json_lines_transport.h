/**
 * @file
 * @brief Declares newline-delimited JSON transport over C++ streams.
 */

#pragma once

#include "transport/transport.h"

#include <iosfwd>
#include <mutex>

namespace atlas::interactive {

/// Exchanges one JSON request or response per stream line.
class JsonLinesTransport final : public Transport {
public:
    /// Borrows input and output streams for the transport lifetime.
    JsonLinesTransport(std::istream& input, std::ostream& output) noexcept;

    /// @copydoc Transport::receive
    bool receive(Request& request, Response& error) override;
    /// @copydoc Transport::send
    void send(const Response& response) override;

private:
    std::istream* _input; ///< Borrowed request stream.
    std::ostream* _output; ///< Borrowed response stream.
    std::mutex _output_mutex; ///< Preserves one-line response framing across writers.
};

}
