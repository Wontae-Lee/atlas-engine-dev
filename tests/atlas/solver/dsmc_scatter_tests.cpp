#include <atlas/solver/dsmc/kernel/dsmc_scatter.h>

#include <atlas/material/atom.h>
#include <atlas/material/material.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using atlas::Atom;
using atlas::Float3;
using atlas::Material;
using atlas::default_random_engine;

/**
 * Scatter only reads each partner's mass and the explicit scattering parameter, so unit-scale
 * masses keep the conservation arithmetic well conditioned while still exercising the mass split.
 */
Material
particle(const float mass) {
    return Material(Atom(mass, 0.0f, 0.0f, 0.0f, 1.0e-10f, 273.0f, 0.5f, 1.0f));
}

/** Mass-weighted total momentum of the pair. */
Float3
momentum(const Float3& lhs_velocity, const Material& lhs, const Float3& rhs_velocity, const Material& rhs) {
    return lhs_velocity * lhs.mass() + rhs_velocity * rhs.mass();
}

/** Twice the kinetic energy of the pair (the constant 1/2 is irrelevant to conservation). */
float
energy(const Float3& lhs_velocity, const Material& lhs, const Float3& rhs_velocity, const Material& rhs) {
    return lhs.mass() * lhs_velocity.length_squared() + rhs.mass() * rhs_velocity.length_squared();
}

/**
 * Recovers the sampled deflection cosine from a scatter.
 *
 * The post-collision relative velocity is the pre-collision one rotated by chi about itself,
 * with the magnitude preserved, so the normalized dot product is exactly `cos(chi)`.
 */
float
deflection_cosine(const Float3& relative_before, const Float3& relative_after) {
    return relative_before.dot(relative_after) / (relative_before.length() * relative_after.length());
}

/** Fixed engine seed, so every statistical assertion below is deterministic. */
constexpr default_random_engine::result_type kSeed = 20240607u;

}

TEST(DsmcScatter, ConservesMomentumEnergyAndRelativeSpeed) {
    const Material lhs = particle(1.0f);
    const Material rhs = particle(3.0f);

    Float3 lhs_velocity(200.0f, -50.0f, 30.0f);
    Float3 rhs_velocity(-80.0f, 120.0f, -10.0f);

    const Float3 lhs_before   = lhs_velocity;
    const Float3 rhs_before   = rhs_velocity;
    const Float3 momentum_pre = momentum(lhs_velocity, lhs, rhs_velocity, rhs);
    const float  energy_pre   = energy(lhs_velocity, lhs, rhs_velocity, rhs);
    const float  speed_pre    = (lhs_velocity - rhs_velocity).length();

    default_random_engine engine(kSeed);

    atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, 1.0f, engine);

    EXPECT_TRUE(atlas::isfinite(lhs_velocity));
    EXPECT_TRUE(atlas::isfinite(rhs_velocity));

    // A non-degenerate collision must actually redirect the pair, not silently pass it through.
    EXPECT_TRUE(lhs_velocity != lhs_before || rhs_velocity != rhs_before);

    const Float3 momentum_post = momentum(lhs_velocity, lhs, rhs_velocity, rhs);
    const float  energy_post   = energy(lhs_velocity, lhs, rhs_velocity, rhs);
    const float  speed_post    = (lhs_velocity - rhs_velocity).length();

    EXPECT_NEAR(momentum_post.x, momentum_pre.x, 1.0e-3f * std::abs(momentum_pre.x) + 1.0e-2f);
    EXPECT_NEAR(momentum_post.y, momentum_pre.y, 1.0e-3f * std::abs(momentum_pre.y) + 1.0e-2f);
    EXPECT_NEAR(momentum_post.z, momentum_pre.z, 1.0e-3f * std::abs(momentum_pre.z) + 1.0e-2f);

    EXPECT_NEAR(energy_post, energy_pre, 1.0e-3f * energy_pre);
    EXPECT_NEAR(speed_post, speed_pre, 1.0e-3f * speed_pre);
}

TEST(DsmcScatter, ConservesInvariantsUnderAnisotropicScattering) {
    const Material lhs = particle(2.0f);
    const Material rhs = particle(5.0f);

    Float3 lhs_velocity(-140.0f, 60.0f, 220.0f);
    Float3 rhs_velocity(90.0f, -170.0f, 40.0f);

    const Float3 momentum_pre = momentum(lhs_velocity, lhs, rhs_velocity, rhs);
    const float  energy_pre   = energy(lhs_velocity, lhs, rhs_velocity, rhs);
    const float  speed_pre    = (lhs_velocity - rhs_velocity).length();

    default_random_engine engine(kSeed);

    // A VSS exponent other than 1 exercises the soft-sphere angular branch (the pow path).
    atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, 1.5f, engine);

    const Float3 momentum_post = momentum(lhs_velocity, lhs, rhs_velocity, rhs);
    const float  energy_post   = energy(lhs_velocity, lhs, rhs_velocity, rhs);
    const float  speed_post    = (lhs_velocity - rhs_velocity).length();

    EXPECT_NEAR(momentum_post.x, momentum_pre.x, 1.0e-3f * std::abs(momentum_pre.x) + 1.0e-2f);
    EXPECT_NEAR(momentum_post.y, momentum_pre.y, 1.0e-3f * std::abs(momentum_pre.y) + 1.0e-2f);
    EXPECT_NEAR(momentum_post.z, momentum_pre.z, 1.0e-3f * std::abs(momentum_pre.z) + 1.0e-2f);

    EXPECT_NEAR(energy_post, energy_pre, 1.0e-3f * energy_pre);
    EXPECT_NEAR(speed_post, speed_pre, 1.0e-3f * speed_pre);
}

TEST(DsmcScatter, PreservesTheCentreOfMassVelocity) {
    const Material lhs = particle(1.0f);
    const Material rhs = particle(4.0f);

    Float3 lhs_velocity(310.0f, -90.0f, 45.0f);
    Float3 rhs_velocity(-25.0f, 160.0f, -75.0f);

    const float  mass_sum = lhs.mass() + rhs.mass();
    const Float3 center_pre
        = (lhs_velocity * lhs.mass() + rhs_velocity * rhs.mass()) / mass_sum;

    default_random_engine engine(kSeed);

    atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, 1.0f, engine);

    const Float3 center_post
        = (lhs_velocity * lhs.mass() + rhs_velocity * rhs.mass()) / mass_sum;

    EXPECT_NEAR(center_post.x, center_pre.x, 1.0e-3f * std::abs(center_pre.x) + 1.0e-3f);
    EXPECT_NEAR(center_post.y, center_pre.y, 1.0e-3f * std::abs(center_pre.y) + 1.0e-3f);
    EXPECT_NEAR(center_post.z, center_pre.z, 1.0e-3f * std::abs(center_pre.z) + 1.0e-3f);
}

TEST(DsmcScatter, IsReproducibleForAnIdenticallySeededEngine) {
    const Material lhs = particle(1.0f);
    const Material rhs = particle(4.0f);

    const Float3 lhs_start(15.0f, -25.0f, 35.0f);
    const Float3 rhs_start(-45.0f, 55.0f, -65.0f);

    Float3                lhs_first = lhs_start;
    Float3                rhs_first = rhs_start;
    default_random_engine first_engine(kSeed);
    atlas::dsmc_scatter(lhs_first, rhs_first, lhs, rhs, 1.0f, first_engine);

    Float3                lhs_second = lhs_start;
    Float3                rhs_second = rhs_start;
    default_random_engine second_engine(kSeed);
    atlas::dsmc_scatter(lhs_second, rhs_second, lhs, rhs, 1.0f, second_engine);

    // Reproducibility now comes from the seed alone, not from a hash of the pair's state.
    EXPECT_FLOAT_EQ(lhs_first.x, lhs_second.x);
    EXPECT_FLOAT_EQ(lhs_first.y, lhs_second.y);
    EXPECT_FLOAT_EQ(lhs_first.z, lhs_second.z);
    EXPECT_FLOAT_EQ(rhs_first.x, rhs_second.x);
    EXPECT_FLOAT_EQ(rhs_first.y, rhs_second.y);
    EXPECT_FLOAT_EQ(rhs_first.z, rhs_second.z);
}

TEST(DsmcScatter, DifferentSeedsDeflectTheSamePairDifferently) {
    const Material lhs = particle(1.0f);
    const Material rhs = particle(4.0f);

    const Float3 lhs_start(15.0f, -25.0f, 35.0f);
    const Float3 rhs_start(-45.0f, 55.0f, -65.0f);

    Float3                lhs_first = lhs_start;
    Float3                rhs_first = rhs_start;
    default_random_engine first_engine(kSeed);
    atlas::dsmc_scatter(lhs_first, rhs_first, lhs, rhs, 1.0f, first_engine);

    Float3                lhs_second = lhs_start;
    Float3                rhs_second = rhs_start;
    default_random_engine second_engine(kSeed + 1u);
    atlas::dsmc_scatter(lhs_second, rhs_second, lhs, rhs, 1.0f, second_engine);

    // The old geometry hash forced one fixed outcome per pre-collision state; a seeded stream
    // must not.
    EXPECT_TRUE(lhs_first != lhs_second || rhs_first != rhs_second);
}

TEST(DsmcScatter, DeflectionAngleIsIndependentOfTheCollisionGeometry) {
    const Material lhs = particle(1.0f);
    const Material rhs = particle(1.0f);

    // Two collisions that share nothing but their seed: different speeds, different directions.
    Float3                slow_lhs(1.0f, 0.0f, 0.0f);
    Float3                slow_rhs(-1.0f, 0.0f, 0.0f);
    const Float3          slow_relative_pre = slow_lhs - slow_rhs;
    default_random_engine slow_engine(kSeed);
    atlas::dsmc_scatter(slow_lhs, slow_rhs, lhs, rhs, 1.0f, slow_engine);

    Float3                fast_lhs(0.0f, 4000.0f, 250.0f);
    Float3                fast_rhs(0.0f, -1500.0f, -700.0f);
    const Float3          fast_relative_pre = fast_lhs - fast_rhs;
    default_random_engine fast_engine(kSeed);
    atlas::dsmc_scatter(fast_lhs, fast_rhs, lhs, rhs, 1.0f, fast_engine);

    const float slow_cosine = deflection_cosine(slow_relative_pre, slow_lhs - slow_rhs);
    const float fast_cosine = deflection_cosine(fast_relative_pre, fast_lhs - fast_rhs);

    // This is the property the RNG buys: the sampled deflection depends on the stream, not on
    // the pair's velocities. The previous sine hash keyed on those velocities, so its angle
    // drifted with their magnitude.
    EXPECT_NEAR(slow_cosine, fast_cosine, 1.0e-4f);
}

TEST(DsmcScatter, IsotropicScatterSpreadsTheDeflectionCosineUniformly) {
    const Material lhs = particle(1.0f);
    const Material rhs = particle(1.0f);

    const Float3 lhs_start(150.0f, -40.0f, 20.0f);
    const Float3 rhs_start(-60.0f, 110.0f, -35.0f);
    const Float3 relative_pre = lhs_start - rhs_start;

    constexpr int   kSamples = 40000;
    constexpr int   kBins    = 10;
    constexpr float kBinned  = static_cast<float>(kSamples) / static_cast<float>(kBins);

    int   histogram[kBins] = {};
    float cosine_sum       = 0.0f;

    // One fixed geometry, one advancing engine: the distribution comes from the stream. A
    // velocity-keyed hash would have returned the same angle all 40000 times.
    default_random_engine engine(kSeed);

    for (int sample = 0; sample < kSamples; ++sample) {
        Float3 lhs_velocity = lhs_start;
        Float3 rhs_velocity = rhs_start;

        atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, 1.0f, engine);

        const float cosine = deflection_cosine(relative_pre, lhs_velocity - rhs_velocity);

        cosine_sum += cosine;

        int bin = static_cast<int>((cosine + 1.0f) * 0.5f * static_cast<float>(kBins));

        bin = (bin < 0) ? 0 : ((bin >= kBins) ? kBins - 1 : bin);

        ++histogram[bin];
    }

    // Isotropic scatter draws cos(chi) uniformly on [-1, 1], so the mean vanishes. The standard
    // error over 40000 draws is ~0.003; 0.02 leaves several sigma of margin.
    EXPECT_NEAR(cosine_sum / static_cast<float>(kSamples), 0.0f, 0.02f);

    for (const int count : histogram) {
        EXPECT_NEAR(static_cast<float>(count), kBinned, 0.15f * kBinned);
    }
}

TEST(DsmcScatter, AnisotropicScatterMatchesTheSoftSphereMean) {
    const Material lhs = particle(1.0f);
    const Material rhs = particle(1.0f);

    const Float3 lhs_start(150.0f, -40.0f, 20.0f);
    const Float3 rhs_start(-60.0f, 110.0f, -35.0f);
    const Float3 relative_pre = lhs_start - rhs_start;

    constexpr int kSamples = 40000;

    // cos(chi) = 2*u^(1/alpha) - 1 with u uniform gives E[cos chi] = (alpha - 1) / (alpha + 1).
    for (const float alpha : { 0.5f, 2.0f }) {
        default_random_engine engine(kSeed);

        float cosine_sum = 0.0f;

        for (int sample = 0; sample < kSamples; ++sample) {
            Float3 lhs_velocity = lhs_start;
            Float3 rhs_velocity = rhs_start;

            atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, alpha, engine);

            cosine_sum += deflection_cosine(relative_pre, lhs_velocity - rhs_velocity);
        }

        const float expected = (alpha - 1.0f) / (alpha + 1.0f);

        EXPECT_NEAR(cosine_sum / static_cast<float>(kSamples), expected, 0.02f);
    }
}

TEST(DsmcScatter, ConsumesExactlyTwoVariatesPerScatter) {
    const Material lhs = particle(1.0f);
    const Material rhs = particle(2.0f);

    Float3 lhs_velocity(120.0f, -30.0f, 15.0f);
    Float3 rhs_velocity(-40.0f, 80.0f, -25.0f);

    default_random_engine scattered(kSeed);
    default_random_engine reference(kSeed);

    atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, 1.0f, scattered);

    // One draw for the deflection cosine, one for the azimuth.
    reference.discard(2);

    EXPECT_EQ(scattered(), reference());
}

TEST(DsmcScatter, IsNoOpWhenRelativeVelocityVanishes) {
    const Material lhs = particle(1.0f);
    const Material rhs = particle(2.0f);

    const Float3 shared(70.0f, -30.0f, 10.0f);
    Float3       lhs_velocity = shared;
    Float3       rhs_velocity = shared;

    default_random_engine engine(kSeed);
    default_random_engine untouched(kSeed);

    atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, 1.0f, engine);

    EXPECT_EQ(lhs_velocity, shared);
    EXPECT_EQ(rhs_velocity, shared);

    // A rejected pair must not steal variates from the stream.
    EXPECT_EQ(engine(), untouched());
}

TEST(DsmcScatter, IsNoOpForNonPositiveScatteringParameter) {
    const Material lhs = particle(1.0f);
    const Material rhs = particle(2.0f);

    const Float3 lhs_start(100.0f, 0.0f, 0.0f);
    const Float3 rhs_start(0.0f, 100.0f, 0.0f);
    Float3       lhs_velocity = lhs_start;
    Float3       rhs_velocity = rhs_start;

    default_random_engine engine(kSeed);
    default_random_engine untouched(kSeed);

    atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, 0.0f, engine);

    EXPECT_EQ(lhs_velocity, lhs_start);
    EXPECT_EQ(rhs_velocity, rhs_start);

    EXPECT_EQ(engine(), untouched());
}

TEST(DsmcScatter, IsNoOpForMasslessPartner) {
    const Material lhs = particle(0.0f);
    const Material rhs = particle(2.0f);

    const Float3 lhs_start(100.0f, 0.0f, 0.0f);
    const Float3 rhs_start(0.0f, 100.0f, 0.0f);
    Float3       lhs_velocity = lhs_start;
    Float3       rhs_velocity = rhs_start;

    default_random_engine engine(kSeed);
    default_random_engine untouched(kSeed);

    atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, 1.0f, engine);

    EXPECT_EQ(lhs_velocity, lhs_start);
    EXPECT_EQ(rhs_velocity, rhs_start);

    EXPECT_EQ(engine(), untouched());
}
