#pragma once

namespace atlas {

/**
 * @brief Hemisphere-sampling law used for the diffuse part of a wall reflection.
 *
 * Selects how `IsothermalCollider::reflect` draws the outgoing direction when a
 * particle is scattered diffusely (i.e. when the momentum accommodation
 * coefficient wins the specular/diffuse coin flip). Both variants sample a
 * direction in the hemisphere about the surface normal; they differ only in the
 * angular density.
 */
enum class DiffuseSampling {

    /**
     * Lambertian (cosine-weighted) hemisphere sampling: the outgoing direction
     * is drawn with probability proportional to the cosine of the angle from the
     * normal, so emission concentrates toward the normal. This is the physically
     * conventional choice for a fully diffuse (thermalizing) wall.
     */
    cosine_weighted,

    /**
     * Uniform hemisphere sampling: every solid-angle direction about the normal
     * is equally likely. Cheaper and flatter than the cosine law; used as the
     * default here for simplicity in the engine's large-domain regime.
     */
    uniform
};

}