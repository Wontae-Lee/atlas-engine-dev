#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/generator/generate_payload.h>

#include <type_traits>

namespace atlas {

enum class GenerateType : int {
    uniform,
    jittering,
    maxwell_sigma,
    maxwell_boltzmann
};

template <typename T>
struct GenerateOperator final {
    GenerateType type = GenerateType::uniform;

    union {
        UniformGenerateOperator<T> uniform;
        JitteringGenerateOperator<T> jittering;
        MaxwellSigmaGenerateOperator<T> maxwell_sigma;
        MaxwellBoltzmannGenerateOperator<T> maxwell_boltzmann;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator(GenerateType type,
                     unsigned int seed = atlas::DEFAULT_UNSIGNED_INT_SEED) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    GenerateOperator(const GenerateOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE GenerateOperator&
    operator=(const GenerateOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~GenerateOperator() noexcept;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, GenerateOperator>, int> = 0>
    ATLAS_HOST GenerateOperator(const Payload& op);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(T param0,
             T param1 = T(1)) const;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    generate(unsigned int seed,
             T param0,
             T param1 = T(1)) const;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    reseed(unsigned int seed) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const GenerateOperator& other) noexcept;
};

}

#if defined(ATLAS_TASKING_CUDA)
namespace thrust {

template <typename T>
struct proclaim_trivially_relocatable<atlas::GenerateOperator<T>> : true_type { };

}
#endif

#include <atlas/generator/generate_operator.hpp>