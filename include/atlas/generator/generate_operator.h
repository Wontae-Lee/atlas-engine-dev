#pragma once

#include <type_traits>

namespace atlas::system {

enum class GenerateType : int {
    maxwell_sigma,
    maxwell_boltzmann,
    uniform
};

}

#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/generator/maxwell_sigma_generator.h>
#include <atlas/generator/uniform_generator.h>

namespace atlas::system {

template <typename T>
struct GenerateOperator final {
    static_assert(std::is_floating_point_v<T>, "GenerateOperator requires a floating-point T");

    GenerateType type = GenerateType::uniform;

    union {

        UniformGenerateOperator<T> uniform;

        MaxwellSigmaGenerateOperator<T> maxwell_sigma;

        MaxwellBoltzmannGenerateOperator<T> maxwell_boltzmann;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator(GenerateType type, unsigned int seed = 0u) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator(const GenerateOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE GenerateOperator&
    operator=(const GenerateOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~GenerateOperator() noexcept;

    ATLAS_HOST
    GenerateOperator(const UniformGenerateOperator<T>& op);

    ATLAS_HOST
    GenerateOperator(const MaxwellSigmaGenerateOperator<T>& op);

    ATLAS_HOST
    GenerateOperator(const MaxwellBoltzmannGenerateOperator<T>& op);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T param0,
             T param1 = T(1)) const;

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const GenerateOperator& other) noexcept;
};

}

namespace atlas {

template <typename T>
using GenerateOperator = atlas::system::GenerateOperator<T>;

using GenerateType = atlas::system::GenerateType;

}

#include <atlas/generator/generate_operator.hpp>