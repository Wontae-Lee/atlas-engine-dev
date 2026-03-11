#pragma once

#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>

namespace atlas::system {

/**
 * @brief Base class representing a physical "matter" concept with an associated solver mass.
 *
 * @details
 * `Matter<T>` is a lightweight base type used by higher-level material models such as
 * @ref Element and @ref Compound. Its primary purpose is to provide a common field
 * (`mass`) representing the mass value used by the simulation/solver.
 *
 *
 * The class is designed to:
 * - remain trivially small (single scalar field),
 * - work well in header-only / template-heavy code,
 * - support a fluent builder pattern consistent with the rest of the codebase.
 *
 * @tparam T Scalar floating-point type (typically `float` or `double`).
 */
template <typename T>
class Matter {
public:
    /**
     * @brief Fluent builder for constructing a @ref Matter instance.
     *
     * @note
     * The builder exists mainly for API consistency with other atlas types
     * (e.g., Domain, Element, Compound, Codecs). In many cases, direct construction
     * via @ref Matter(T) is also perfectly fine.
     */
    class Builder;

    /**
     * @brief Mass used by the solver.
     *
     * @details
     * This value is interpreted by derived classes and the simulation logic.
     * The default value is `T(1)`.
     */
    T mass = T(1);

    /// @brief Default constructor. Initializes @ref mass to `T(1)`.
    Matter() = default;

    /// @brief Virtual destructor for safe polymorphic usage.
    virtual ~Matter() = default;

    /**
     * @brief Construct with an explicit solver mass.
     *
     * @param mass_ Solver mass to store in @ref mass.
     *
     * @pre `mass_` should be finite and non-negative (recommended).
     *
     * @note
     * This constructor does not perform heavy validation by itself; any strict
     * checks should be implemented in user code or in the builder's validation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Matter(T mass_);

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;
};

/**
 * @brief Builder for @ref Matter.
 *
 * @details
 * The builder stores configuration (currently only mass), validates it, and can
 * construct the object either:
 * - by value via @ref build(), or
 * - as a host shared pointer via @ref make_host_shared().
 *
 * Typical usage:
 * @code
 * auto m = atlas::system::Matter<double>::builder()
 *              .with_mass(1.0e-6)
 *              .build();
 * @endcode
 *
 * @tparam T Scalar floating-point type.
 */
template <typename T>
class Matter<T>::Builder final {
public:
    /// @brief Construct an empty builder (mass defaults to `T(1)`).
    Builder() = default;

    /**
     * @brief Set the solver mass.
     *
     * @param mass New mass value to store.
     * @return `*this` for fluent chaining.
     *
     * @note
     * The mass should generally be finite and non-negative. The builder may
     * enforce this in @ref validate_or_throw().
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_mass(T mass) noexcept;

    /**
     * @brief Build a @ref Matter object by value.
     *
     * @return Constructed @ref Matter instance.
     *
     * @throws std::invalid_argument if configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Matter<T>
    build() const;

    /**
     * @brief Build a @ref Matter object on the host as a shared pointer.
     *
     * @return Host shared pointer to a constructed @ref Matter instance.
     *
     * @throws std::invalid_argument if configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Matter<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validate builder state and throw if invalid.
     *
     * @details
     * Implementations typically validate constraints such as:
     * - mass must be finite
     * - mass must be non-negative
     *
     * The specific rules are determined by the project’s conventions.
     */
    void
    validate_or_throw() const;

private:
    /// @brief Pending mass value to be used during construction.
    T _mass = T(1);
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Public alias for @ref atlas::system::Matter.
 *
 * @tparam T Scalar floating-point type.
 */
template <typename T>
using Matter = system::Matter<T>;

/// @brief Host shared pointer alias for @ref Matter.
template <typename T>
using MatterHostPtr = atlas::host_shared_ptr<system::Matter<T>>;

/// @brief Device shared pointer alias for @ref Matter.
template <typename T>
using MatterDevicePtr = atlas::device_shared_ptr<system::Matter<T>>;

} // namespace atlas

#include <atlas/matter/matter.hpp>
