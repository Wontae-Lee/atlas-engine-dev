#include <atlas/solver/dsmc/kernel/hard_sphere_kernel.h>

#include <atlas/material/atom.h>
#include <atlas/material/material.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace {

using atlas::Atom;
using atlas::Float3;
using atlas::HardSphereKernel;
using atlas::Material;

/** The leaf is a stateless POD, so it must be copyable into the device union and default-usable. */
static_assert(std::is_trivially_copyable_v<HardSphereKernel>);
static_assert(std::is_default_constructible_v<HardSphereKernel>);

/**
 * @brief Build an atomic species; only mass and reference diameter matter for hard spheres.
 * The viscosity index and scattering parameter are carried but unused by this kernel.
 */
Material
species(const float mass, const float reference_diameter) {
    return Material(Atom(mass, 0.0f, 0.0f, 0.0f, reference_diameter, 273.0f, 0.75f, 1.0f));
}

/** Mass-weighted total momentum of the pair. */
Float3
momentum(const Float3& lhs_velocity, const Material& lhs, const Float3& rhs_velocity, const Material& rhs) {
    return lhs_velocity * lhs.mass() + rhs_velocity * rhs.mass();
}

/** Twice the pair kinetic energy (the 1/2 factor is irrelevant to conservation). */
float
kinetic_energy(const Float3& lhs_velocity, const Material& lhs, const Float3& rhs_velocity, const Material& rhs) {
    return lhs.mass() * lhs_velocity.length_squared() + rhs.mass() * rhs_velocity.length_squared();
}

/**
 * @brief Scatter the pair once, drawing from @p engine, and return the cosine of the
 *        deflection angle chi.
 *
 * The post-collision relative velocity equals the rotated pre-collision relative velocity, so the
 * angle between them is the sampled scattering angle. The pair velocities are taken by value so a
 * caller can hold one fixed geometry and re-scatter it repeatedly, advancing a single engine to
 * sweep out the angular distribution.
 */
float
deflection_cosine(const HardSphereKernel& kernel,
                  const Material& lhs,
                  const Material& rhs,
                  Float3 lhs_velocity,
                  Float3 rhs_velocity,
                  atlas::default_random_engine& engine) {
    const Float3 relative_before = lhs_velocity - rhs_velocity;
    kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine);
    const Float3 relative_after = lhs_velocity - rhs_velocity;
    return relative_before.dot(relative_after) / (relative_before.length() * relative_after.length());
}

}

TEST(HardSphereKernel, DefaultConstructsAndEvaluates) {
    const HardSphereKernel kernel {};

    const float cross_section = HardSphereKernel::cross_section(species(1.0f, 3.0e-10f), species(1.0f, 3.0e-10f), 100.0f);

    EXPECT_TRUE(std::isfinite(cross_section));
    EXPECT_GT(cross_section, 0.0f);
}

TEST(HardSphereKernel, CrossSectionIsPiTimesMeanDiameterSquared) {
    // Unequal diameters exercise the arithmetic-mean rule d = (d_lhs + d_rhs) / 2.
    const Material lhs = species(1.0f, 3.0e-10f);
    const Material rhs = species(1.0f, 5.0e-10f);

    const float mean_diameter = (3.0e-10f + 5.0e-10f) * 0.5f;
    const float expected       = atlas::pi * mean_diameter * mean_diameter;

    const float cross_section = HardSphereKernel::cross_section(lhs, rhs, 250.0f);

    EXPECT_NEAR(cross_section, expected, expected * 1.0e-5f);
}

TEST(HardSphereKernel, CrossSectionIsSpeedIndependent) {
    // The defining property of the hard-sphere model: sigma does not vary with relative speed.
    const Material lhs = species(1.0f, 4.0e-10f);
    const Material rhs = species(1.0f, 4.0e-10f);

    const float slow = HardSphereKernel::cross_section(lhs, rhs, 1.0e-3f);
    const float fast = HardSphereKernel::cross_section(lhs, rhs, 1.0e4f);

    EXPECT_FLOAT_EQ(slow, fast);
}

TEST(HardSphereKernel, CrossSectionIsZeroForNonPositiveMeanDiameter) {
    // A degenerate (zero-diameter) pair carries no collision area.
    EXPECT_FLOAT_EQ(HardSphereKernel::cross_section(species(1.0f, 0.0f), species(1.0f, 0.0f), 100.0f), 0.0f);
}

TEST(HardSphereKernel, ConservesMomentumAndEnergyForEqualMasses) {
    const Material lhs = species(2.0f, 3.0e-10f);
    const Material rhs = species(2.0f, 3.0e-10f);

    const HardSphereKernel kernel {};

    Float3 lhs_velocity(150.0f, -60.0f, 45.0f);
    Float3 rhs_velocity(-90.0f, 70.0f, -25.0f);

    const Float3 momentum_pre = momentum(lhs_velocity, lhs, rhs_velocity, rhs);
    const float  energy_pre   = kinetic_energy(lhs_velocity, lhs, rhs_velocity, rhs);

    atlas::default_random_engine engine(11u);
    kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine);

    const Float3 momentum_post = momentum(lhs_velocity, lhs, rhs_velocity, rhs);
    const float  energy_post   = kinetic_energy(lhs_velocity, lhs, rhs_velocity, rhs);

    EXPECT_TRUE(atlas::isfinite(lhs_velocity));
    EXPECT_TRUE(atlas::isfinite(rhs_velocity));
    EXPECT_NEAR(momentum_post.x, momentum_pre.x, 1.0e-3f * std::abs(momentum_pre.x) + 1.0e-2f);
    EXPECT_NEAR(momentum_post.y, momentum_pre.y, 1.0e-3f * std::abs(momentum_pre.y) + 1.0e-2f);
    EXPECT_NEAR(momentum_post.z, momentum_pre.z, 1.0e-3f * std::abs(momentum_pre.z) + 1.0e-2f);
    EXPECT_NEAR(energy_post, energy_pre, 1.0e-3f * energy_pre);
}

TEST(HardSphereKernel, ConservesMomentumAndEnergyForUnequalMasses) {
    const Material lhs = species(1.0f, 3.0e-10f);
    const Material rhs = species(6.0f, 4.0e-10f);

    const HardSphereKernel kernel {};

    Float3 lhs_velocity(220.0f, -30.0f, 15.0f);
    Float3 rhs_velocity(-40.0f, 110.0f, -80.0f);

    const Float3 momentum_pre = momentum(lhs_velocity, lhs, rhs_velocity, rhs);
    const float  energy_pre   = kinetic_energy(lhs_velocity, lhs, rhs_velocity, rhs);

    atlas::default_random_engine engine(22u);
    kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine);

    const Float3 momentum_post = momentum(lhs_velocity, lhs, rhs_velocity, rhs);
    const float  energy_post   = kinetic_energy(lhs_velocity, lhs, rhs_velocity, rhs);

    EXPECT_TRUE(atlas::isfinite(lhs_velocity));
    EXPECT_TRUE(atlas::isfinite(rhs_velocity));
    EXPECT_NEAR(momentum_post.x, momentum_pre.x, 1.0e-3f * std::abs(momentum_pre.x) + 1.0e-2f);
    EXPECT_NEAR(momentum_post.y, momentum_pre.y, 1.0e-3f * std::abs(momentum_pre.y) + 1.0e-2f);
    EXPECT_NEAR(momentum_post.z, momentum_pre.z, 1.0e-3f * std::abs(momentum_pre.z) + 1.0e-2f);
    EXPECT_NEAR(energy_post, energy_pre, 1.0e-3f * energy_pre);
}

TEST(HardSphereKernel, PreservesRelativeSpeedAndCenterOfMass) {
    const Material lhs = species(1.0f, 3.0e-10f);
    const Material rhs = species(4.0f, 3.0e-10f);

    const HardSphereKernel kernel {};

    Float3 lhs_velocity(180.0f, -90.0f, 60.0f);
    Float3 rhs_velocity(-70.0f, 130.0f, -20.0f);

    const float  speed_pre  = (lhs_velocity - rhs_velocity).length();
    const Float3 center_pre = momentum(lhs_velocity, lhs, rhs_velocity, rhs) / (lhs.mass() + rhs.mass());

    atlas::default_random_engine engine(33u);
    kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine);

    const float  speed_post  = (lhs_velocity - rhs_velocity).length();
    const Float3 center_post = momentum(lhs_velocity, lhs, rhs_velocity, rhs) / (lhs.mass() + rhs.mass());

    EXPECT_NEAR(speed_post, speed_pre, 1.0e-3f * speed_pre);
    EXPECT_NEAR(center_post.x, center_pre.x, 1.0e-3f * std::abs(center_pre.x) + 1.0e-2f);
    EXPECT_NEAR(center_post.y, center_pre.y, 1.0e-3f * std::abs(center_pre.y) + 1.0e-2f);
    EXPECT_NEAR(center_post.z, center_pre.z, 1.0e-3f * std::abs(center_pre.z) + 1.0e-2f);
}

TEST(HardSphereKernel, ScatterIsIsotropic) {
    // Hard-sphere scatter draws cos(chi) uniformly on [-1, 1]. Hold ONE fixed pre-collision
    // geometry and re-scatter it repeatedly from a single seeded engine: the angular law comes
    // entirely from the RNG, so the recovered deflection cosines must fill every bin roughly
    // evenly. This catches a velocity-dependent bias the old geometry-hash test could not, since
    // the input geometry never changes. The seed makes the histogram deterministic, not flaky.
    const Material lhs = species(1.0f, 3.0e-10f);
    const Material rhs = species(3.0f, 3.0e-10f);

    const HardSphereKernel kernel {};

    const Float3 lhs_velocity(180.0f, -95.0f, 60.0f);
    const Float3 rhs_velocity(-70.0f, 125.0f, -25.0f);

    constexpr int sample_count = 40000;
    constexpr int bin_count    = 10;

    std::array<int, bin_count> histogram {};
    double                     cosine_sum = 0.0;

    atlas::default_random_engine engine(20260710u);

    for (int i = 0; i < sample_count; ++i) {
        const float cosine = deflection_cosine(kernel, lhs, rhs, lhs_velocity, rhs_velocity, engine);

        cosine_sum += cosine;

        int bin = static_cast<int>((cosine + 1.0f) * 0.5f * bin_count);
        bin     = (bin < 0) ? 0 : (bin >= bin_count ? bin_count - 1 : bin);
        ++histogram[static_cast<std::size_t>(bin)];
    }

    const double expected_per_bin = static_cast<double>(sample_count) / bin_count;

    for (const int count : histogram) {
        EXPECT_GT(count, expected_per_bin * 0.5) << "a bin is underpopulated: scatter is not isotropic";
        EXPECT_LT(count, expected_per_bin * 1.5) << "a bin is overpopulated: scatter is not isotropic";
    }

    // A symmetric [-1, 1] distribution has zero mean cosine.
    EXPECT_NEAR(cosine_sum / sample_count, 0.0, 0.05);
}

TEST(HardSphereKernel, ScatterIsReproducibleForIdenticalSeeds) {
    // Scatter is no longer a function of the pair's geometry; reproducibility now comes solely
    // from the engine's seed. Identical input plus two identically seeded engines must agree
    // bit-for-bit.
    const Material lhs = species(1.0f, 3.0e-10f);
    const Material rhs = species(4.0f, 3.0e-10f);

    const HardSphereKernel kernel {};

    const Float3 lhs_start(35.0f, -55.0f, 75.0f);
    const Float3 rhs_start(-45.0f, 65.0f, -85.0f);

    atlas::default_random_engine engine_first(4242u);
    atlas::default_random_engine engine_second(4242u);

    Float3 lhs_first = lhs_start;
    Float3 rhs_first = rhs_start;
    kernel(lhs_first, rhs_first, lhs, rhs, engine_first);

    Float3 lhs_second = lhs_start;
    Float3 rhs_second = rhs_start;
    kernel(lhs_second, rhs_second, lhs, rhs, engine_second);

    EXPECT_FLOAT_EQ(lhs_first.x, lhs_second.x);
    EXPECT_FLOAT_EQ(lhs_first.y, lhs_second.y);
    EXPECT_FLOAT_EQ(lhs_first.z, lhs_second.z);
    EXPECT_FLOAT_EQ(rhs_first.x, rhs_second.x);
    EXPECT_FLOAT_EQ(rhs_first.y, rhs_second.y);
    EXPECT_FLOAT_EQ(rhs_first.z, rhs_second.z);
}

TEST(HardSphereKernel, ScatterDiffersForDifferentSeeds) {
    // The flip side of reproducibility: with the geometry hash gone, differently seeded engines
    // draw different deflection angles, so the same input pair scatters to a different state.
    const Material lhs = species(1.0f, 3.0e-10f);
    const Material rhs = species(4.0f, 3.0e-10f);

    const HardSphereKernel kernel {};

    const Float3 lhs_start(35.0f, -55.0f, 75.0f);
    const Float3 rhs_start(-45.0f, 65.0f, -85.0f);

    atlas::default_random_engine engine_first(4242u);
    atlas::default_random_engine engine_second(9001u);

    Float3 lhs_first = lhs_start;
    Float3 rhs_first = rhs_start;
    kernel(lhs_first, rhs_first, lhs, rhs, engine_first);

    Float3 lhs_second = lhs_start;
    Float3 rhs_second = rhs_start;
    kernel(lhs_second, rhs_second, lhs, rhs, engine_second);

    const bool differs = lhs_first.x != lhs_second.x || lhs_first.y != lhs_second.y
        || lhs_first.z != lhs_second.z || rhs_first.x != rhs_second.x
        || rhs_first.y != rhs_second.y || rhs_first.z != rhs_second.z;

    EXPECT_TRUE(differs) << "two distinct seeds produced an identical scatter";
}

TEST(HardSphereKernel, DegeneratePairDoesNotConsumeVariates) {
    // A pair with zero relative speed hits the early return, which must leave the engine
    // untouched. Two identically seeded engines therefore stay in lockstep: the first is never
    // advanced, so its next draw equals the second's.
    const Material lhs = species(2.0f, 3.0e-10f);
    const Material rhs = species(2.0f, 3.0e-10f);

    const HardSphereKernel kernel {};

    // Equal velocities -> zero relative speed -> degenerate scatter.
    Float3 lhs_velocity(50.0f, -20.0f, 10.0f);
    Float3 rhs_velocity(50.0f, -20.0f, 10.0f);

    atlas::default_random_engine engine_a(13579u);
    atlas::default_random_engine engine_b(13579u);

    kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine_a);

    EXPECT_EQ(engine_a(), engine_b());
}

TEST(HardSphereKernel, SuccessfulScatterAdvancesEngineExactlyTwice) {
    // A successful scatter draws exactly two variates: the deflection cosine and the azimuth.
    // Discarding two from a twin engine must leave both in the same state.
    const Material lhs = species(1.0f, 3.0e-10f);
    const Material rhs = species(4.0f, 3.0e-10f);

    const HardSphereKernel kernel {};

    Float3 lhs_velocity(120.0f, -40.0f, 25.0f);
    Float3 rhs_velocity(-60.0f, 80.0f, -15.0f);

    atlas::default_random_engine engine_a(24680u);
    atlas::default_random_engine engine_b(24680u);

    kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine_a);
    engine_b.discard(2);

    EXPECT_EQ(engine_a(), engine_b());
}
