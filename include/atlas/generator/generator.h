#pragma once

#include <atlas/generator/generate.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

namespace atlas {

class Generator {
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

    ATLAS_NODISCARD ATLAS_HOST virtual Float3
    generate() const = 0;

    ATLAS_NODISCARD ATLAS_HOST virtual const Generate&
    generate_operator() const noexcept = 0;

    ATLAS_NODISCARD ATLAS_HOST virtual Generate
    make_generate_operator() const noexcept = 0;

    ATLAS_NODISCARD ATLAS_HOST virtual float
    param0() const noexcept = 0;

    ATLAS_NODISCARD ATLAS_HOST virtual float
    param1() const noexcept = 0;

    ATLAS_NODISCARD ATLAS_HOST virtual GenerateType
    type() const noexcept = 0;
};

using GeneratorHostPtr = atlas::host_shared_ptr<Generator>;

using GeneratorDevicePtr = atlas::device_shared_ptr<Generator>;

}
