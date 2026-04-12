#pragma once

/**
 * @file generator.h
 * @brief Declares the abstract particle-velocity generator interface.
 *
 * Generators produce particle velocities on the host side and can export a
 * backend-portable GenerateOperator that Source uses during device emission.
 */

#include <atlas/generator/generate_operator.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Abstract base class for host-side particle generators.
 *
 * Concrete generators expose their runtime parameters, produce sample
 * velocities on the host, and provide a GenerateOperator suitable for
 * backend-side emission code.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
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

    ATLAS_HOST ATLAS_NODISCARD virtual const GenerateOperator<T>&
    generate_operator() const noexcept = 0;

    ATLAS_HOST ATLAS_NODISCARD virtual GenerateOperator<T>
    make_generate_operator() const noexcept = 0;

    ATLAS_HOST ATLAS_NODISCARD virtual T
    param0() const noexcept = 0;

    ATLAS_HOST ATLAS_NODISCARD virtual T
    param1() const noexcept = 0;

    ATLAS_HOST ATLAS_NODISCARD virtual GenerateType
    type() const noexcept = 0;
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::system::Generator.
 */
template <typename T>
using Generator = atlas::system::Generator<T>;

template <typename T>
using GeneratorHostPtr = atlas::host_shared_ptr<atlas::system::Generator<T>>;

template <typename T>
using GeneratorDevicePtr = atlas::device_shared_ptr<atlas::system::Generator<T>>;

}

#include <atlas/generator/generator.hpp>
