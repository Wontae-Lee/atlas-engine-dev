#pragma once

namespace atlas::interactive::transport {

enum class Command {
    initialize,
    start,
    pause,
    step,
    close,
    shutdown
};

}
