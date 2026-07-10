#pragma once

namespace atlas {

/**
 * @brief Discriminator tag for the @c Generator tagged-union umbrella.
 *
 * Selects which self-contained leaf a @c Generator currently holds and drives
 * dispatch through @c GeneratorVariant (the @c HostVariant instantiation). The
 * underlying type is fixed to @c int so the value is stable across the
 * host/device boundary and cheap to store next to the union.
 *
 * @note The enumerator order must stay in lockstep with the @c HostVariantCase
 *       list in @c generator.h; each case maps a tag to a union member.
 */
enum class GeneratorType : int {

    /// Uniform velocity sampling within [min_value, max_value] plus a bulk offset.
    uniform,

    /// A constant base velocity perturbed by a small uniform jitter, plus bulk.
    jittering,

    /// Isotropic Gaussian velocity with a caller-supplied standard deviation.
    maxwell_sigma,

    /// Maxwell-Boltzmann velocity whose sigma is derived from temperature and mass.
    maxwell_boltzmann
};

}