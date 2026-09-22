#pragma once

#include <atlas/system/system.h>

#include <cstddef>
#include <optional>

namespace atlas::interactive {

class Session final {
public:
    enum class State {
        empty,
        ready,
        running,
        paused
    };

    Session() = default;
    explicit Session(System system);

    void initialize(System system);
    void start();
    void pause();
    void update();
    void step(std::size_t count = 1);
    void close();

    bool initialized() const noexcept;
    bool running() const noexcept;
    State state() const noexcept;

    System& system();
    const System& system() const;

private:
    std::optional<System> _system;
    State _state = State::empty;
};

}
