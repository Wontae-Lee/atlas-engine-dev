#pragma once

/**
 * @file generator.h
 * @brief Declares the abstract host-side particle generator interface and related aliases.
 *
 * @details
 * This header defines @ref atlas::system::Generator, the abstract base class
 * used to represent host-side particle-velocity generation laws in Atlas.
 *
 * A generator is responsible for:
 * - producing particle-velocity samples on the host,
 * - exposing its runtime parameters in a uniform way,
 * - reporting its generator type,
 * - providing a backend-portable @ref GenerateOperator that can be consumed by
 *   device-side emission code.
 *
 * ## Host/device split
 * Atlas separates particle generation into two complementary representations:
 * - the **host-side polymorphic generator**, represented by @ref Generator,
 * - the **device-friendly tagged-union operator**, represented by
 *   @ref GenerateOperator.
 *
 * This allows applications to configure generators through normal C++ host-side
 * abstractions while still supporting efficient device-side emission paths.
 *
 * ## Typical workflow
 * A concrete generator implementation generally supports the following flow:
 * 1. The application configures a generator instance on the host.
 * 2. The generator can immediately produce host-side samples through
 *    @ref generate.
 * 3. The generator can expose or create a backend-portable
 *    @ref GenerateOperator through @ref generate_operator or
 *    @ref make_generate_operator.
 * 4. Runtime systems such as particle sources can use that operator in backend
 *    kernels for device-side particle emission.
 *
 * ## Parameter model
 * Each concrete generator exposes two scalar parameters through:
 * - @ref param0
 * - @ref param1
 *
 * Their exact physical meaning depends on the concrete generator type. They may
 * represent quantities such as:
 * - thermal scale,
 * - variance or sigma,
 * - uniform bounds,
 * - mass- or temperature-related inputs.
 *
 * ## Ownership model
 * This interface is intended for polymorphic use through shared pointers. The
 * namespace aliases at the bottom of this header provide convenient host/device
 * pointer types for that purpose.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for generated velocities and generator parameters.
 */

#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/random/seed.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Forward declaration of the runtime generator type discriminator.
 *
 * @details
 * The concrete definition lives elsewhere and identifies which backend-portable
 * generation law is associated with a generator.
 */
enum class GenerateType : int;

template <typename T>
struct GenerateOperator;

/**
 * @brief Abstract base class for host-side particle generators.
 *
 * @details
 * @ref Generator defines the common interface implemented by all host-side
 * particle-velocity generation laws in Atlas.
 *
 * Concrete derived classes are expected to:
 * - hold generator-specific runtime parameters,
 * - produce host-side velocity samples,
 * - expose a backend-portable @ref GenerateOperator,
 * - report their concrete generation law through @ref type.
 *
 * ## Responsibilities
 * A generator implementation typically provides:
 * - one-shot host-side sample generation through @ref generate,
 * - cached or referenced access to an internal generate operator through
 *   @ref generate_operator,
 * - by-value operator export through @ref make_generate_operator,
 * - parameter introspection through @ref param0 and @ref param1.
 *
 * ## Why this interface exists
 * This abstraction allows Atlas to:
 * - configure emission laws polymorphically on the host,
 * - preserve a common interface across multiple distribution families,
 * - convert those generators into compact backend-side operators for runtime
 *   emission.
 *
 * ## Copy and move semantics
 * The base class provides defaulted copy and move operations so derived
 * generators can participate in normal value and pointer-based workflows as
 * long as their own state supports those operations.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Generator {
    static_assert(std::is_floating_point_v<T>, "Generator requires a floating-point T");

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a generator base with default-initialized state.
     */
    Generator() = default;

    /**
     * @brief Virtual destructor.
     *
     * @details
     * Declared virtual so concrete generators can be destroyed correctly through
     * base-class pointers.
     */
    virtual ~Generator() = default;

    /**
     * @brief Defaulted copy constructor.
     *
     * @details
     * Allows derived generators to participate in copy-based workflows.
     */
    Generator(const Generator&) = default;

    /**
     * @brief Defaulted copy assignment operator.
     *
     * @return `*this`.
     */
    Generator&
    operator=(const Generator&)
        = default;

    /**
     * @brief Defaulted move constructor.
     *
     * @details
     * Allows derived generators to participate in move-based workflows.
     */
    Generator(Generator&&) = default;

    /**
     * @brief Defaulted move assignment operator.
     *
     * @return `*this`.
     */
    Generator&
    operator=(Generator&&)
        = default;

    /**
     * @brief Generate a particle velocity sample on the host.
     *
     * @details
     * This pure virtual function produces a velocity sample according to the
     * concrete generator law implemented by the derived class.
     *
     * @return Generated velocity vector.
     */
    ATLAS_HOST ATLAS_NODISCARD virtual Vector3<T>
    generate() const = 0;

    /**
     * @brief Return a reference to the backend-portable generate operator.
     *
     * @details
     * Provides access to a device-friendly @ref GenerateOperator associated with
     * this generator instance.
     *
     * Implementations commonly return a reference to an internally cached
     * operator object whose lifetime is tied to the generator instance.
     *
     * @return Const reference to the associated generate operator.
     */
    ATLAS_HOST ATLAS_NODISCARD virtual const GenerateOperator<T>&
    generate_operator() const noexcept = 0;

    /**
     * @brief Create a backend-portable generate operator by value.
     *
     * @details
     * Returns a by-value @ref GenerateOperator representing this generator's
     * concrete generation law and runtime parameters.
     *
     * This is useful when a caller needs an independent backend-portable copy
     * rather than a reference to an internal cached operator.
     *
     * @return Generate operator corresponding to this generator.
     */
    ATLAS_HOST ATLAS_NODISCARD virtual GenerateOperator<T>
    make_generate_operator() const noexcept = 0;

    /**
     * @brief Return the primary scalar parameter of the generator.
     *
     * @details
     * The exact meaning of this parameter is generator-specific.
     *
     * @return Primary scalar parameter.
     */
    ATLAS_HOST ATLAS_NODISCARD virtual T
    param0() const noexcept = 0;

    /**
     * @brief Return the secondary scalar parameter of the generator.
     *
     * @details
     * The exact meaning of this parameter is generator-specific.
     *
     * @return Secondary scalar parameter.
     */
    ATLAS_HOST ATLAS_NODISCARD virtual T
    param1() const noexcept = 0;

    /**
     * @brief Return the runtime type tag of the concrete generator.
     *
     * @details
     * Identifies which backend-portable generation law is associated with this
     * concrete generator instance.
     *
     * @return Corresponding @ref GenerateType value.
     */
    ATLAS_HOST ATLAS_NODISCARD virtual GenerateType
    type() const noexcept = 0;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::Generator.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Generator = atlas::system::Generator<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to
 *        @ref atlas::system::Generator.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using GeneratorHostPtr = atlas::host_shared_ptr<atlas::system::Generator<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to
 *        @ref atlas::system::Generator.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using GeneratorDevicePtr = atlas::device_shared_ptr<atlas::system::Generator<T>>;

} // namespace atlas

#include <atlas/generator/generator.hpp>
