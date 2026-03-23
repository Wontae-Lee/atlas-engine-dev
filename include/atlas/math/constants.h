#pragma once
#include <cmath>
#include <limits>

namespace atlas {

namespace math {

    constexpr double k_epsilon_d = 1e-12;

    constexpr float k_epsilon_f = 1e-6f;

    constexpr double k_farthest_d = 1e30;

    constexpr float k_farthest_f = 1e30f;
}

constexpr double pi = M_PI;

constexpr double boltzmann_constant = 1.380649e-23;

constexpr double eps = math::k_epsilon_d;

constexpr double far = math::k_farthest_d;

constexpr double inf = std::numeric_limits<double>::infinity();

constexpr double tol = eps;

}