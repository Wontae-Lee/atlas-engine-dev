#pragma once

#include <atlas/generator/generate.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

/**
 * @file generator.h
 * @brief Host-only virtual interface for a per-species velocity
 *        generator configuration, the host-side counterpart to the
 *        device-callable `Generate`.
 *
 * @details
 * Concrete generators (`UniformGenerator`, `MaxwellBoltzmannGenerator`,
 * ...) hold whatever host-side parameters their sampling model needs
 * (min/max, temperature, bulk velocity, ...) and expose
 * `make_generate_operator()` to produce the matching device-callable
 * `Generate` value — the same host-interface/device-value split
 * used by `Geometry`/`Geometry` (see
 * `docs/architecture/04-backend-portability.md` §4.7): virtual dispatch
 * only exists host-side, since device code cannot resolve it without
 * `-rdc`. `Fluid::generators()` holds one `Generator` per species;
 * `Source`/`Source` read the matching `Generate`
 * per particle's assigned species.
 */

namespace atlas {

/**
 * @brief Host-only interface for a species' velocity generator
 *        configuration. See this file's top-of-file documentation for
 *        the host/device split this mirrors.
 */
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

    /** @brief Draws one sample host-side, using this generator's own
     *  internal stateful engine. */
    ATLAS_NODISCARD ATLAS_HOST virtual Float3
    generate() const = 0;

    /** @brief The `Generate` value this generator owns
     *  (constructed once, reused). */
    ATLAS_NODISCARD ATLAS_HOST virtual const Generate&
    generate_operator() const noexcept = 0;

    /** @brief Constructs a fresh `Generate` value matching this
     *  generator's configuration, for callers that need their own copy
     *  (e.g. to embed into a per-species device array). */
    ATLAS_NODISCARD ATLAS_HOST virtual Generate
    make_generate_operator() const noexcept = 0;

    /** @brief The first parameter this generator's `Generate`
     *  expects (meaning depends on `type()`; see `generate.h`). */
    ATLAS_NODISCARD ATLAS_HOST virtual float
    param0() const noexcept = 0;

    /** @brief The second parameter (meaning depends on `type()`). */
    ATLAS_NODISCARD ATLAS_HOST virtual float
    param1() const noexcept = 0;

    /** @brief Which `GenerateType` this generator wraps. */
    ATLAS_NODISCARD ATLAS_HOST virtual GenerateType
    type() const noexcept = 0;
};

using GeneratorHostPtr = atlas::host_shared_ptr<Generator>;

using GeneratorDevicePtr = atlas::device_shared_ptr<Generator>;

}
