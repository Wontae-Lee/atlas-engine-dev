#pragma once

#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/random/seed.h>

#include <type_traits>

namespace atlas::fluid {

enum class GenerateType : int;

template <typename T>
struct GenerateOperator;

template <typename T>
class Generator {
    static_assert(std::is_floating_point_v<T>, "Generator requires a floating-point T");

public:
    Generator() = default;

    virtual ~Generator() = default;

    Generator(const Generator&) = default;

    Generator&
    operator=(const Generator&)
        = default;

    Generator(Generator&&) = default;

    Generator&
    operator=(Generator&&)
        = default;

    ATLAS_HOST ATLAS_NODISCARD virtual Vector3<T>
    generate() const = 0;

    ATLAS_HOST ATLAS_NODISCARD virtual const GenerateOperator<T>&
    generate_operator() const noexcept
        = 0;

    ATLAS_HOST ATLAS_NODISCARD virtual GenerateOperator<T>
    make_generate_operator() const noexcept = 0;

    ATLAS_HOST ATLAS_NODISCARD virtual T
    param0() const noexcept
        = 0;

    ATLAS_HOST ATLAS_NODISCARD virtual T
    param1() const noexcept
        = 0;

    ATLAS_HOST ATLAS_NODISCARD virtual GenerateType
    type() const noexcept
        = 0;
};

}

namespace atlas {

template <typename T>
using Generator = atlas::fluid::Generator<T>;

template <typename T>
using GeneratorHostPtr = atlas::host_shared_ptr<atlas::fluid::Generator<T>>;

template <typename T>
using GeneratorDevicePtr = atlas::device_shared_ptr<atlas::fluid::Generator<T>>;

}

#include <atlas/generator/generator.hpp>