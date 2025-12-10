#pragma once
namespace atlas {
namespace math {
    constexpr double k_epsilon_d  = 1e-12;
    constexpr float k_epsilon_f   = 1e-6f;
    constexpr double k_farthest_d = 1e30;
    constexpr float k_farthest_f  = 1e30f;

}
constexpr double pi  = M_PI;
constexpr float eps  = math::k_epsilon_f;
constexpr float far  = math::k_farthest_f;
constexpr float inf  = std::numeric_limits<float>::infinity();
constexpr double tol = 1e-8;
}