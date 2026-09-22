#include "session/session.h"

#include <stdexcept>
#include <utility>

namespace atlas::interactive {

Session::Session(System system) {
    initialize(std::move(system));
}

void
Session::initialize(System system) {
    _system.emplace(std::move(system));
    _state = State::ready;
}

void
Session::start() {
    if (!_system) throw std::logic_error("Session must be initialized before it starts.");
    _state = State::running;
}

void
Session::pause() {
    if (!_system) throw std::logic_error("Session must be initialized before it pauses.");
    _state = State::paused;
}

void
Session::update() {
    if (_state == State::running) _system->update();
}

void
Session::step(const std::size_t count) {
    if (!_system) throw std::logic_error("Session must be initialized before it steps.");
    for (std::size_t index = 0; index < count; ++index) _system->update();
}

void
Session::close() {
    _system.reset();
    _state = State::empty;
}

bool
Session::initialized() const noexcept {
    return _system.has_value();
}

bool
Session::running() const noexcept {
    return _state == State::running;
}

Session::State
Session::state() const noexcept {
    return _state;
}

System&
Session::system() {
    if (!_system) throw std::logic_error("Session does not own a System.");
    return *_system;
}

const System&
Session::system() const {
    if (!_system) throw std::logic_error("Session does not own a System.");
    return *_system;
}

}
