#pragma once

/**
 * @file uniform_generator.h
 * @brief Declares uniform particle-velocity generation operators and host-side generator types.
 *
 * @details
 * This header defines:
 * - @ref atlas::UniformGenerateOperator, a backend-portable operator for
 *   sampling velocity vectors with independently uniform components, and
 * - @ref atlas::UniformGenerator, a host-side polymorphic generator that
 *   owns the corresponding runtime parameters and exports a portable
 *   @ref GenerateOperator.
 *
 * ## Distribution model
 * The uniform generator samples each velocity component independently from a
 * uniform distribution:
 * \f[
 * v_i \sim \mathcal{U}(\text{min}, \text{max})
 * \f]
 * for each axis \f$i \in \{x,y,z\}\f$.
 *
 * This is useful for:
 * - randomized initialization,
 * - bounded velocity injection,
 * - testing and debugging emission pipelines.
 *
 * ## Host/device split
 * Atlas separates this generation law into:
 * - a host-side @ref UniformGenerator used for configuration and direct sampling,
 * - a backend-friendly @ref UniformGenerateOperator embedded in
 *   @ref GenerateOperator for device-side emission.
 *
 * ## Construction
 * The host-side generator may be:
 * - constructed directly from minimum and maximum bounds, or
 * - built through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for parameters and generated velocities.
 */

#include <atlas/math/math.h>

#include <optional>

#include <atlas/generator/generator.h>
#include <atlas/generator/generate_operator.h>

namespace atlas {

template <typename T>
class UniformGenerator final : public Generator<T> {
public:
    class Builder;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    UniformGenerator(
        T min_value,
        T max_value,
        unsigned int seed = atlas::DEFAULT_UNSIGNED_INT_SEED) noexcept;

    ATLAS_HOST ATLAS_NODISCARD Vector3<T>
    generate() const override;

    ATLAS_HOST ATLAS_NODISCARD const GenerateOperator<T>&
    generate_operator() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD GenerateOperator<T>
    make_generate_operator() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD T
    param0() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD T
    param1() const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD GenerateType
    type() const noexcept override;

private:
    T _min_value;
    T _max_value;
    GenerateOperator<T> _operator;
};

template <typename T>
class UniformGenerator<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_min_value(T min_value) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_max_value(T max_value) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_seed(unsigned int seed) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE UniformGenerator<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<UniformGenerator<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<T> _min_value;
    std::optional<T> _max_value;
    unsigned int _seed = atlas::DEFAULT_UNSIGNED_INT_SEED;
};

} // namespace atlas

#include <atlas/generator/uniform_generator.hpp>
