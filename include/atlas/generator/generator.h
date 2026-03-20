#pragma once

#include <atlas/generator/generate_operator.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Abstract velocity generator interface.
 *
 * @details
 * A `Generator<T>` stores the parameters required to populate a
 * `DeviceBuffer<Vector3<T>>` with generated values and exposes a uniform
 * polymorphic API through @ref generate.
 *
 * @tparam T Floating-point scalar type.
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

    /**
     * @brief Fills `values` according to the concrete generator policy.
     *
     * @param values Output buffer to overwrite.
     */
    ATLAS_HOST virtual void
    generate(DeviceBuffer<Vector3<T>>& values) const = 0;

    /**
     * @brief Returns the runtime generation kind represented by this object.
     *
     * @return Concrete generation type.
     */
    ATLAS_HOST ATLAS_NODISCARD virtual GenerateType
    type() const noexcept = 0;
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using Generator = atlas::system::Generator<T>;

template <typename T>
using GeneratorHostPtr = atlas::host_shared_ptr<atlas::system::Generator<T>>;

template <typename T>
using GeneratorDevicePtr = atlas::device_shared_ptr<atlas::system::Generator<T>>;

} // namespace atlas

#include <atlas/generator/generator.hpp>
