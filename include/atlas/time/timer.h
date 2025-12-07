#ifndef INCLUDE_ATLAS_CORE_TIMER_H
#define INCLUDE_ATLAS_CORE_TIMER_H
#include <atlas/core/macros.h>
#include <chrono>

namespace atlas {
class Timer {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE
    Timer();

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE float
    duration_in_seconds() const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset();

private:
    std::chrono::steady_clock _clock;
    std::chrono::steady_clock::time_point _startingPoint;
};
}
#endif