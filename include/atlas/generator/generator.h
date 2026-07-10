#pragma once

#include <atlas/core/host_variant.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator_type.h>
#include <atlas/generator/jittering_generator.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/generator/maxwell_sigma_generator.h>
#include <atlas/generator/uniform_generator.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <concepts>
#include <cstddef>
#include <utility>

namespace atlas {

/**
 * @brief Compile-time contract every generator leaf must satisfy.
 *
 * A conforming leaf produces the initial velocity and species of freshly
 * spawned particles. It must expose two host-callable operations: a @c const
 * @c generate that writes into the fluid state and reports how many slots it
 * filled, and a mutable @c set_bulk_velocity that retargets the drift added to
 * every sampled velocity. The @c static_assert block below enforces this for
 * each concrete leaf, so a leaf added to the union without the right signatures
 * fails to compile here rather than at the dispatch site.
 *
 * @tparam G Candidate generator leaf type.
 */
template <typename G>
concept ConceptGenerator = requires(G generator,
                                    const G const_generator,
                                    FluidVelocityState* velocities,
                                    FluidSpeciesState* species,
                                    const Float3 bulk_velocity,
                                    std::size_t offset,
                                    std::size_t count) {
    { const_generator.generate(velocities, species, offset, count) } -> std::same_as<int>;
    { generator.set_bulk_velocity(bulk_velocity) } -> std::same_as<void>;
};

static_assert(ConceptGenerator<UniformGenerator>);
static_assert(ConceptGenerator<JitteringGenerator>);
static_assert(ConceptGenerator<MaxwellSigmaGenerator>);
static_assert(ConceptGenerator<MaxwellBoltzmannGenerator>);

/**
 * @brief Tagged-union umbrella over the four particle-emission velocity models.
 *
 * A @c Generator owns exactly one leaf at a time and forwards @c generate /
 * @c set_bulk_velocity to it through @c GeneratorVariant. Each leaf holds one or
 * more @c DeviceBuffer members (species ratios, numbers, masses) which have a
 * host-only copy, so this umbrella is move-only and its lifetime is driven by
 * the @c HostVariant helpers rather than a compiler-generated union path.
 *
 * The engine pairs one generator with one source: after a source spawns
 * particles at a contiguous range, the matching generator fills that same range
 * with initial velocities and species ids (see @c System::spawn_particles).
 *
 * @note Host-only. The buffers live on the device but the dispatch machinery
 *       runs on the host; the per-particle sampling happens inside the leaf's
 *       device kernel.
 */
class Generator final {
public:
    /// Active leaf discriminator; must agree with the union member in use.
    GeneratorType type = GeneratorType::uniform;

    /**
     * @brief Storage for the active leaf; exactly one member is alive at a time.
     *
     * Which member is live is dictated by @c type. Construction, move, and
     * destruction of the anonymous union are all routed through
     * @c GeneratorVariant so the correct member's special functions run.
     */
    union {

        /// Live when @c type == GeneratorType::uniform.
        UniformGenerator uniform;

        /// Live when @c type == GeneratorType::jittering.
        JitteringGenerator jittering;

        /// Live when @c type == GeneratorType::maxwell_sigma.
        MaxwellSigmaGenerator maxwell_sigma;

        /// Live when @c type == GeneratorType::maxwell_boltzmann.
        MaxwellBoltzmannGenerator maxwell_boltzmann;
    };

    /**
     * @brief Constructs a default umbrella holding a default @c UniformGenerator.
     *
     * The leaf is value-initialized with empty species buffers, so calling
     * @c generate before a real leaf is installed produces zero particles.
     */
    ATLAS_HOST
    Generator() noexcept;

    /**
     * @brief Wraps a fully built @c UniformGenerator, taking ownership of it.
     * @param op Leaf to move into the union; left in a moved-from state.
     */
    ATLAS_HOST explicit Generator(UniformGenerator op) noexcept;

    /**
     * @brief Wraps a fully built @c JitteringGenerator, taking ownership of it.
     * @param op Leaf to move into the union; left in a moved-from state.
     */
    ATLAS_HOST explicit Generator(JitteringGenerator op) noexcept;

    /**
     * @brief Wraps a fully built @c MaxwellSigmaGenerator, taking ownership of it.
     * @param op Leaf to move into the union; left in a moved-from state.
     */
    ATLAS_HOST explicit Generator(MaxwellSigmaGenerator op) noexcept;

    /**
     * @brief Wraps a fully built @c MaxwellBoltzmannGenerator, taking ownership.
     * @param op Leaf to move into the union; left in a moved-from state.
     */
    ATLAS_HOST explicit Generator(MaxwellBoltzmannGenerator op) noexcept;

    /// Deleted: leaves own device buffers with a host-only copy, so no copy.
    Generator(const Generator&) = delete;

    /// Deleted: see the deleted copy constructor.
    Generator&
    operator=(const Generator&)
        = delete;

    /**
     * @brief Move-constructs from @p other, transferring the active leaf.
     * @param other Source umbrella; left holding a moved-from leaf of same tag.
     */
    ATLAS_HOST
    Generator(Generator&& other) noexcept;

    /**
     * @brief Move-assigns from @p other, destroying the current leaf first.
     * @param other Source umbrella; left holding a moved-from leaf of same tag.
     * @return Reference to @c *this.
     */
    ATLAS_HOST Generator&
    operator=(Generator&& other) noexcept;

    /// Destroys the active leaf via the variant's tag-driven destroy path.
    ATLAS_HOST
    ~Generator() noexcept;

    /**
     * @brief Fills a contiguous range of the fluid state with sampled velocities
     *        and species ids by dispatching to the active leaf.
     *
     * Writes to @c [offset, offset + count) of the velocity and species buffers.
     * Each leaf independently clamps the range to the smaller of the two buffer
     * capacities and skips out-of-range or empty requests, so the returned count
     * may be less than @p count.
     *
     * @param velocities Target velocity state; a null pointer yields 0.
     * @param species    Target species state; a null pointer yields 0.
     * @param offset     First slot to write, typically the current particle count.
     * @param count      Number of slots requested, typically the spawn count.
     * @return Number of slots actually written, or 0 if the tag matches no leaf.
     */
    ATLAS_NODISCARD ATLAS_HOST int
    generate(FluidVelocityState* velocities,
             FluidSpeciesState* species,
             std::size_t offset,
             std::size_t count) const;

    /**
     * @brief Retargets the constant drift added to every sampled velocity.
     * @param bulk_velocity New bulk (drift) velocity, in the fluid's units.
     */
    ATLAS_HOST void
    set_bulk_velocity(const Float3& bulk_velocity) noexcept;
};

/**
 * @brief @c HostVariant binding that maps each @c GeneratorType to its member.
 *
 * Provides the tag-driven construct / move / destroy / visit / apply helpers the
 * umbrella forwards to. The default tag is @c GeneratorType::uniform, matching
 * the union member the default constructor installs.
 */
using GeneratorVariant = HostVariant<
    Generator,
    GeneratorType,
    GeneratorType::uniform,
    HostVariantCase<GeneratorType::uniform, &Generator::uniform>,
    HostVariantCase<GeneratorType::jittering, &Generator::jittering>,
    HostVariantCase<GeneratorType::maxwell_sigma, &Generator::maxwell_sigma>,
    HostVariantCase<GeneratorType::maxwell_boltzmann, &Generator::maxwell_boltzmann>>;

/**
 * @brief Const visitor that forwards @c generate to whichever leaf is active.
 *
 * Bundles the four @c generate arguments so a single templated @c operator()
 * can be applied to any leaf type by @c GeneratorVariant::visit.
 */
class GeneratorGenerate {
public:
    FluidVelocityState* velocities; ///< Target velocity state, forwarded verbatim.
    FluidSpeciesState* species;     ///< Target species state, forwarded verbatim.
    std::size_t offset;             ///< First slot to write.
    std::size_t count;              ///< Number of slots requested.
    /**
     * @brief Invokes @c generate on a concrete leaf.
     * @tparam G Deduced leaf type.
     * @param generator The active leaf.
     * @return Slots the leaf actually filled.
     */
    template <typename G>
    ATLAS_HOST int
    operator()(const G& generator) const {
        return generator.generate(velocities, species, offset, count);
    }
};

/**
 * @brief Mutable visitor that forwards @c set_bulk_velocity to the active leaf.
 */
class GeneratorSetBulkVelocity {
public:
    const Float3& bulk_velocity; ///< New drift velocity, bound by reference.
    /**
     * @brief Updates the bulk velocity of a concrete leaf in place.
     * @tparam G Deduced leaf type.
     * @param generator The active leaf.
     */
    template <typename G>
    ATLAS_HOST void
    operator()(G& generator) const { generator.set_bulk_velocity(bulk_velocity); }
};

// Each definition below is a thin forwarder into GeneratorVariant so that the
// union's special member functions run against whichever leaf the tag selects;
// the umbrella itself holds no logic of its own.

ATLAS_HOST ATLAS_FORCE_INLINE
Generator::Generator() noexcept {
    GeneratorVariant::construct(*this, GeneratorType::uniform);
}

ATLAS_HOST ATLAS_FORCE_INLINE
Generator::Generator(UniformGenerator op) noexcept {
    GeneratorVariant::construct_payload(*this, std::move(op));
}

ATLAS_HOST ATLAS_FORCE_INLINE
Generator::Generator(JitteringGenerator op) noexcept {
    GeneratorVariant::construct_payload(*this, std::move(op));
}

ATLAS_HOST ATLAS_FORCE_INLINE
Generator::Generator(MaxwellSigmaGenerator op) noexcept {
    GeneratorVariant::construct_payload(*this, std::move(op));
}

ATLAS_HOST ATLAS_FORCE_INLINE
Generator::Generator(MaxwellBoltzmannGenerator op) noexcept {
    GeneratorVariant::construct_payload(*this, std::move(op));
}

ATLAS_HOST ATLAS_FORCE_INLINE
Generator::Generator(Generator&& other) noexcept {
    GeneratorVariant::move_construct(*this, std::move(other));
}

ATLAS_HOST ATLAS_FORCE_INLINE Generator&
Generator::operator=(Generator&& other) noexcept {
    GeneratorVariant::move_assign(*this, std::move(other));
    return *this;
}

ATLAS_HOST ATLAS_FORCE_INLINE
    Generator::~Generator() noexcept {
    GeneratorVariant::destroy(*this);
}

ATLAS_HOST ATLAS_FORCE_INLINE int
Generator::generate(FluidVelocityState* velocities,
                    FluidSpeciesState* species,
                    const std::size_t offset,
                    const std::size_t count) const {
    // The trailing 0 is the fallback visit() returns when type matches no case.
    return GeneratorVariant::visit(
        *this,
        GeneratorGenerate { velocities, species, offset, count },
        0);
}

ATLAS_HOST ATLAS_FORCE_INLINE void
Generator::set_bulk_velocity(const Float3& bulk_velocity) noexcept {
    GeneratorVariant::apply(*this, GeneratorSetBulkVelocity { bulk_velocity });
}

/// Shared owning handle to a host-resident @c Generator umbrella.
using GeneratorHostPtr = atlas::host_shared_ptr<Generator>;

/// Shared handle to a device-resident @c Generator (device-side ownership).
using GeneratorDevicePtr = atlas::device_shared_ptr<Generator>;

}