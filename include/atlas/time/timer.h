#pragma once
#include <atlas/core/macros.h>
#include <chrono>

namespace atlas {
class Timer {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE

    Timer() {
        _starting_point = std::chrono::steady_clock::now();
    }

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE

        float
        duration_in_seconds() const {
        const auto end   = std::chrono::steady_clock::now();
        const auto count = std::chrono::duration_cast<std::chrono::microseconds>(end - _starting_point).count();
        return static_cast<float>(count) / 1000000.0f;
    }

    ATLAS_HOST ATLAS_FORCE_INLINE

        void
        reset() {
        _starting_point = std::chrono::steady_clock::now();
    }

private:
    std::chrono::steady_clock _clock;
    std::chrono::steady_clock::time_point _starting_point;
};
}