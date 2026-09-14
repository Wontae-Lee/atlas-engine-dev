#include "register.h"

#include <atlas/math/vector/float3.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/sampling/sampling.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

void
register_sampling(nb::module_& m) {
    nb::class_<default_random_engine>(m, "DefaultRandomEngine")
        .def(nb::init<default_random_engine::result_type>(),
             "seed"_a = default_random_engine::default_seed)
        .def(nb::init<const default_random_engine&>(), "other"_a)
        .def("__call__", &default_random_engine::operator())
        .def("seed", &default_random_engine::seed, "seed"_a)
        .def("discard", &default_random_engine::discard, "count"_a)
        .def_static("min", &default_random_engine::min)
        .def_static("max", &default_random_engine::max)
        .def_ro_static("multiplier", &default_random_engine::multiplier)
        .def_ro_static("modulus", &default_random_engine::modulus)
        .def_ro_static("default_seed", &default_random_engine::default_seed);

    nb::class_<uniform_real_distribution<float>>(m, "UniformRealDistribution")
        .def(nb::init<float, float>(), "min_value"_a = 0.0f, "max_value"_a = 1.0f)
        .def("__call__", &uniform_real_distribution<float>::operator(), "engine"_a)
        .def("min", &uniform_real_distribution<float>::min)
        .def("max", &uniform_real_distribution<float>::max);

    nb::class_<uniform_real_distribution<double>>(m, "UniformRealDistributionDouble")
        .def(nb::init<double, double>(), "min_value"_a = 0.0, "max_value"_a = 1.0)
        .def("__call__", &uniform_real_distribution<double>::operator(), "engine"_a)
        .def("min", &uniform_real_distribution<double>::min)
        .def("max", &uniform_real_distribution<double>::max);

    m.attr("DEFAULT_UNSIGNED_INT_SEED") = nb::int_(atlas::DEFAULT_UNSIGNED_INT_SEED);

    m.def("shuffle_key", &atlas::shuffle_key, "index"_a, "seed"_a);
    m.def("generate_standard_normal_pair", [](default_random_engine& engine) {
        float first {};
        float second {};
        atlas::generate_standard_normal_pair(engine, first, second);
        return nb::make_tuple(first, second);
    }, "engine"_a);
    m.def("generate_standard_normal", &atlas::generate_standard_normal, "engine"_a);
    m.def("sample_uniform_vector", &atlas::sample_uniform_vector,
          "engine"_a, "min_value"_a, "max_value"_a);
    m.def("sample_normal_vector", &atlas::sample_normal_vector, "engine"_a, "sigma"_a);
    m.def("build_orthonormal_basis", [](const Float3& normal) {
        Float3 tangent {};
        Float3 bitangent {};
        atlas::build_orthonormal_basis(normal, tangent, bitangent);
        return nb::make_tuple(tangent, bitangent);
    }, "normal"_a);
    m.def("sample_uniform_hemisphere", &atlas::sample_uniform_hemisphere,
          "normal"_a, "u1"_a, "u2"_a);
    m.def("sample_cosine_hemisphere", &atlas::sample_cosine_hemisphere,
          "normal"_a, "u1"_a, "u2"_a);
    m.def("sample_random_unit_vector", &atlas::sample_random_unit_vector, "engine"_a);
    m.def("sample_directional_unit_vector", &atlas::sample_directional_unit_vector,
          "incoming_direction"_a, "alpha"_a, "engine"_a);
    m.def("sample_axis_count", [](const float lower, const float upper, const float spacing) {
        if (std::isfinite(lower) && std::isfinite(upper) && std::isfinite(spacing)
            && spacing > 0.0f && upper >= lower
            && !((upper - lower) / spacing < static_cast<float>(std::numeric_limits<int>::max()))) {
            throw nb::value_error("sample count exceeds the C++ integer range");
        }
        return atlas::sample_axis_count(lower, upper, spacing);
    }, "lower"_a, "upper"_a, "spacing"_a);
    m.def("sample_hashed_unit_interval", [](const Float3& seed, const float salt) {
        return atlas::sample_hashed_unit_interval(seed, salt);
    }, "seed"_a, "salt"_a);
    m.def("sample_hashed_unit_interval", [](const int index, const std::uint64_t seed) {
        return atlas::sample_hashed_unit_interval(index, seed);
    }, "index"_a, "seed"_a);
    m.def("sample_hashed_index", &atlas::sample_hashed_index,
          "index"_a, "upper_bound"_a, "seed"_a);
    m.def("sample_weighted_index", [](const std::vector<float>& weights, default_random_engine& engine) {
        if (weights.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw nb::value_error("weight count exceeds the C++ integer range");
        }
        return atlas::sample_weighted_index(weights.data(), static_cast<int>(weights.size()), engine);
    }, "weights"_a, "engine"_a,
    "Draws from normalized non-negative weights; an empty table returns zero without drawing.");
    m.def("sample_weighted_choice", [](const std::vector<float>& weights,
                                       const std::vector<float>& values,
                                       default_random_engine& engine) {
        if (weights.size() != values.size()) {
            throw nb::value_error("weights and values must have the same length");
        }
        if (weights.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw nb::value_error("weight count exceeds the C++ integer range");
        }
        const long double limit = std::ldexp(1.0L, std::numeric_limits<std::size_t>::digits);
        for (const float value : values) {
            if (!std::isfinite(value) || value < 0.0f || static_cast<long double>(value) >= limit) {
                throw nb::value_error("values must be finite, non-negative, and representable as size_t");
            }
        }
        return atlas::sample_weighted_choice(weights.data(), values.data(),
                                             static_cast<int>(weights.size()), engine);
    }, "weights"_a, "values"_a, "engine"_a,
    "Draws a value using normalized non-negative weights and truncates it to an integer; "
    "an empty table returns zero without drawing.");
}

}
