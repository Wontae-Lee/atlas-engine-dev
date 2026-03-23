#pragma once

#include <atlas/generator/generate_operator.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class Generator {
    static_assert(std::is_floating_point_v<T>, "Generator requires a floating-point T");

public:
    Generator()          = default;
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

    ATLAS_HOST ATLAS_NODISCARD virtual GenerateType
    type() const noexcept = 0;
};

}

namespace atlas {

template <typename T>
using Generator = atlas::system::Generator<T>;

template <typename T>
using GeneratorHostPtr = atlas::host_shared_ptr<atlas::system::Generator<T>>;

template <typename T>
using GeneratorDevicePtr = atlas::device_shared_ptr<atlas::system::Generator<T>>;

}

#include <atlas/generator/generator.hpp>