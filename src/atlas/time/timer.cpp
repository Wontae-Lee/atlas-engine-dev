#include <atlas/time/timer.h>
using namespace atlas;
Timer::Timer() {
    _startingPoint = std::chrono::steady_clock::now();
}

float
Timer::duration_in_seconds() const {
    const auto end   = std::chrono::steady_clock::now();
    const auto count = std::chrono::duration_cast<std::chrono::microseconds>(end - _startingPoint).count();
    return static_cast<float>(count) / 1000000.0f;
}

void
Timer::reset() {
    _startingPoint = std::chrono::steady_clock::now();
}