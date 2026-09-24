#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/container/type_store.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/memory/memory.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace atlas {

/**
 * @brief Type-erased, keyed collection of the fluid's per-particle attribute columns.
 *
 * A @c TypeStore keyed on the concrete @c FluidState leaf type: at most one instance of
 * each attribute (position, velocity, species, …) can be registered, and each is looked
 * up by its type. Iterating it visits every registered column, which is how @c compact()
 * gathers all attributes with a single index list.
 */
using FluidStateStore = TypeStore<FluidState>;

/**
 * @brief The particle population: a structure-of-arrays gas held on the device.
 *
 * A @c Fluid owns a fixed-capacity set of per-particle attribute columns (a
 * @c FluidStateStore of @c FluidState leaves) plus the bookkeeping buffers used to
 * remove dead particles. Attributes are stored as separate device columns rather than an
 * array-of-structs so kernels touch only the fields they need and so a new attribute can
 * be added without changing existing code. Buffers are allocated once to @c buffer_size
 * (the capacity); @c particle_count tracks how many leading slots are currently alive.
 *
 * Construction seeds the three mandatory columns — position, velocity, species — sized to
 * capacity. Optional columns (temperature, the energy modes) are registered on demand via
 * @c emplace_state / @c set_state.
 *
 * The type is move-only: it owns device buffers whose copy is host-only, so it is passed
 * around as a @c FluidHostPtr. All members are host-side; kernels reach the data through
 * a trivially-copyable view (see @c FluidDsmcView) built from the raw device pointers.
 */
class Fluid final {
public:
    /** @brief Fluent builder for constructing a validated @c Fluid; defined below. */
    class Builder;

    /** @brief Constructs an empty fluid with no capacity and no attribute columns. */
    Fluid() = default;

    /**
     * @brief Constructs a fluid of the given capacity with the mandatory columns.
     *
     * Allocates the active-flag buffer and the position, velocity, and species columns,
     * each sized to @p buffer_size. The live particle count starts at zero.
     *
     * @param buffer_size Maximum number of particles the fluid can hold.
     */
    ATLAS_HOST explicit Fluid(std::size_t buffer_size);

    /**
     * @brief Constructs a fluid of the given capacity and attaches a material dictionary.
     *
     * Delegates to @c Fluid(std::size_t) for the columns, then adopts @p materials, which
     * the species column indexes into for per-particle material properties.
     *
     * @param buffer_size Maximum number of particles the fluid can hold.
     * @param materials   Shared handle to the material dictionary; may be null.
     */
    ATLAS_HOST
    Fluid(std::size_t buffer_size, MaterialDictionaryHostPtr materials);

    /** @brief Deleted: the fluid owns device buffers and is non-copyable. */
    Fluid(const Fluid&) = delete;

    /** @brief Move-constructs, transferring ownership of all columns and buffers. */
    Fluid(Fluid&&) noexcept = default;

    /** @brief Defaulted; device buffers free themselves. */
    ~Fluid() = default;

    /** @brief Deleted: the fluid owns device buffers and is non-copyable. */
    Fluid&
    operator=(const Fluid&)
        = delete;

    /** @brief Move-assigns, transferring ownership of all columns and buffers. */
    Fluid&
    operator=(Fluid&&) noexcept = default;

    /** @brief Returns a fresh @c Builder for fluent construction. */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Sets the number of live particles.
     *
     * @param particle_count New live count; must not exceed @c buffer_size.
     * @throws std::out_of_range if @p particle_count exceeds the fluid's capacity.
     */
    ATLAS_HOST void
    set_particle_count(std::size_t particle_count);

    /**
     * @brief Returns the per-particle survivor-flag buffer (mutable).
     *
     * Element @c i is 1 if particle @c i survives the current step and 0 if it is to be
     * removed. Populated by the removal pass before @c compact() consumes it as the scan
     * input. Sized to capacity.
     */
    ATLAS_NODISCARD ATLAS_HOST DeviceBuffer<int>&
    active() noexcept;

    /** @brief Returns the per-particle survivor-flag buffer (const). @see active() */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<int>&
    active() const noexcept;

    /**
     * @brief Removes dead particles, packing survivors into the leading slots.
     *
     * Reads the @c active() flags for the first @c particle_count particles, prefix-sums
     * them to assign each survivor a compacted slot, builds the survivor index list, and
     * asks every registered @c FluidState to gather itself by that list so all columns
     * stay row-aligned. Finally refills the leading survivor flags with 1 and updates the
     * live particle count. Entirely device-side apart from reading back the survivor
     * total. A no-op that returns the current count if nothing is alive or if all
     * particles survive.
     *
     * @return The number of surviving particles, which becomes the new live count.
     */
    ATLAS_HOST std::size_t
    compact();

    /**
     * @brief Constructs an attribute column of type @p StateT in place, replacing any
     *        existing column of that type.
     *
     * @tparam StateT A concrete @c FluidState leaf type.
     * @tparam Args   Constructor argument types for @p StateT.
     * @param args    Forwarded to the @p StateT constructor (typically the buffer size).
     * @return Reference to the newly constructed state.
     */
    template <typename StateT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE StateT&
    emplace_state(Args&&... args) {
        return _states.template emplace<StateT>(std::forward<Args>(args)...);
    }

    /**
     * @brief Registers a pre-built attribute column, replacing any existing one of that
     *        type.
     *
     * @tparam StateT A concrete @c FluidState leaf type.
     * @param state   Owning pointer to the column; must not be null.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_state(std::unique_ptr<StateT> state) {
        _states.template set<StateT>(std::move(state));
    }

    /**
     * @brief Returns the registered column of type @p StateT, or null if absent.
     * @tparam StateT A concrete @c FluidState leaf type.
     * @return Non-owning pointer to the column, or @c nullptr if not registered.
     */
    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE StateT*
    state() noexcept {
        return _states.template get<StateT>();
    }

    /** @copydoc state() */
    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const StateT*
    state() const noexcept {
        return _states.template get<StateT>();
    }

    /**
     * @brief Reports whether a column of type @p StateT is registered.
     * @tparam StateT A concrete @c FluidState leaf type.
     * @return @c true if the column is present.
     */
    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE bool
    has_state() const noexcept {
        return _states.template contains<StateT>();
    }

    /**
     * @brief Detaches and returns the column of type @p StateT.
     * @tparam StateT A concrete @c FluidState leaf type.
     * @return Owning pointer to the removed column, or @c nullptr if it was absent.
     */
    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE std::unique_ptr<StateT>
    remove_state() {
        return _states.template remove<StateT>();
    }

    /**
     * @brief Builds a trivially-copyable device view over this fluid's columns.
     *
     * Delegates to @c ViewT::make(*this), which gathers the raw device pointers of the
     * columns the view needs into a plain struct a device lambda can capture by value.
     *
     * @tparam ViewT A view type exposing a static @c make(Fluid&) factory (e.g.
     *               @c FluidDsmcView).
     * @return The constructed view by value.
     */
    template <typename ViewT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE ViewT
    view() {
        return ViewT::make(*this);
    }

    /** @brief Returns the store of registered attribute columns (mutable). */
    ATLAS_NODISCARD ATLAS_HOST FluidStateStore&
    states() noexcept;

    /** @brief Returns the store of registered attribute columns (const). */
    ATLAS_NODISCARD ATLAS_HOST const FluidStateStore&
    states() const noexcept;

    /** @brief Returns the allocated capacity (per-column length), in particles. */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    buffer_size() const noexcept;

    /** @brief Returns the current number of live particles (the leading valid slots). */
    ATLAS_NODISCARD ATLAS_HOST std::size_t
    particle_count() const noexcept;

    /**
     * @brief Returns the statistical weight: real molecules represented per simulated
     *        particle.
     *
     * The macroscopic scale factor of the DSMC method; always positive. Used to convert
     * simulated-particle counts and collision rates to physical quantities.
     */
    ATLAS_NODISCARD ATLAS_HOST float
    statistical_weight() const noexcept;

    /** @brief Returns the attached material dictionary; the handle may be null. */
    ATLAS_NODISCARD ATLAS_HOST const MaterialDictionaryHostPtr&
    materials() const noexcept;

private:
    /** @brief Grants the builder access to the private fields it sets after construction. */
    friend class Builder;

    std::size_t _particle_count = 0; ///< Number of live particles in the leading slots.

    std::size_t _buffer_size = 0; ///< Allocated capacity of every column, in particles.

    float _statistical_weight = 1.0f; ///< Real molecules per simulated particle; positive.

    MaterialDictionaryHostPtr _materials {}; ///< Shared material dictionary; may be null.

    FluidStateStore _states; ///< The registered per-particle attribute columns.

    DeviceBuffer<int> _active; ///< Per-particle survivor flags (0/1); scan input for compact().

    DeviceBuffer<int> _survivor_offsets; ///< Scratch: exclusive prefix sum of the flags.

    DeviceBuffer<std::size_t> _compact_indices; ///< Scratch: survivor source indices from the scan.

    DeviceBuffer<int> _survivor_total; ///< Scratch: single-element device holder for the count.
};

/**
 * @brief Fluent builder that collects a fluid's parameters and constructs a validated
 *        @c Fluid.
 *
 * Setters accumulate the capacity, initial live count, statistical weight, material
 * dictionary, and optional initial state prefixes. @c build() validates their
 * relationships and produces the fluid. The mandatory attribute columns are
 * created by the @c Fluid constructor, then initialized by the builder.
 */
class Fluid::Builder final {
public:
    /** @brief Constructs a builder holding the default parameter values. */
    Builder() = default;

    /**
     * @brief Sets the fluid's capacity (per-column allocation length), in particles.
     * @param buffer_size Maximum number of particles.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_buffer_size(std::size_t buffer_size) noexcept;

    /**
     * @brief Sets the initial number of live particles.
     * @param particle_count Initial live count; must not exceed the buffer size.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_particle_count(std::size_t particle_count) noexcept;

    /**
     * @brief Sets the statistical weight (real molecules per simulated particle).
     * @param statistical_weight Scale factor; must be positive.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_statistical_weight(float statistical_weight) noexcept;

    /**
     * @brief Attaches the material dictionary the species column indexes into.
     * @param materials Shared handle to the dictionary; may be null.
     * @return @c *this for chaining.
     */
    ATLAS_HOST Builder&
    with_materials(MaterialDictionaryHostPtr materials) noexcept;

    /** @brief Stage live positions; when supplied, the length must equal particle_count. */
    ATLAS_HOST Builder& with_position(HostBuffer<Float3> values);
    /** @brief Stage live velocities; when supplied, the length must equal particle_count. */
    ATLAS_HOST Builder& with_velocity(HostBuffer<Float3> values);
    /** @brief Stage live species IDs; IDs must index the attached dictionary when present. */
    ATLAS_HOST Builder& with_species(HostBuffer<std::size_t> values);
    /** @brief Stage an optional temperature column for every live particle. */
    ATLAS_HOST Builder& with_temperature(HostBuffer<float> values);
    /** @brief Stage an optional translational-energy column for every live particle. */
    ATLAS_HOST Builder& with_translational_energy(HostBuffer<float> values);
    /** @brief Stage an optional rotational-energy column for every live particle. */
    ATLAS_HOST Builder& with_rotational_energy(HostBuffer<float> values);
    /** @brief Stage an optional vibrational-energy column for every live particle. */
    ATLAS_HOST Builder& with_vibrational_energy(HostBuffer<float> values);

    /**
     * @brief Validates the parameters and constructs the fluid by value.
     * @return The constructed @c Fluid.
     * @throws std::runtime_error if the statistical weight, live count, state lengths,
     *         or material indices are invalid.
     */
    ATLAS_NODISCARD ATLAS_HOST Fluid
    build() const;

    /**
     * @brief Same as @c build() but returns the fluid through a host-owned pointer.
     * @return A @c host_unique_ptr owning the constructed @c Fluid.
     * @throws std::runtime_error under the same conditions as @c build().
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_unique_ptr<Fluid>
    make_host_unique() const;

private:
    /**
     * @brief Throws if the accumulated parameters are inconsistent.
     * @throws std::runtime_error if a fluid parameter, state length, or species ID
     *         is inconsistent.
     */
    ATLAS_HOST void
    validate() const;

private:
    std::size_t _particle_count = 0; ///< Requested initial live particle count.

    std::size_t _buffer_size = 0; ///< Requested capacity, in particles.

    float _statistical_weight = 1.0f; ///< Requested statistical weight; must be positive.

    MaterialDictionaryHostPtr _materials {}; ///< Material dictionary to attach; may be null.

    std::optional<HostBuffer<Float3>> _position; ///< Supplied live positions.
    std::optional<HostBuffer<Float3>> _velocity; ///< Supplied live velocities.
    std::optional<HostBuffer<std::size_t>> _species; ///< Supplied live material IDs.
    std::optional<HostBuffer<float>> _temperature; ///< Supplied live temperatures.
    std::optional<HostBuffer<float>> _translational_energy; ///< Supplied live translational energies.
    std::optional<HostBuffer<float>> _rotational_energy; ///< Supplied live rotational energies.
    std::optional<HostBuffer<float>> _vibrational_energy; ///< Supplied live vibrational energies.
};

/** @brief Host-owning pointer to a @c Fluid, the standard way to pass the move-only type. */
using FluidHostPtr = atlas::host_unique_ptr<Fluid>;

}
