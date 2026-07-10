#include <atlas/solver/dsmc/kernel/variable_soft_sphere_kernel.h>

#include <atlas/material/atom.h>
#include <atlas/material/material.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/solver/dsmc/kernel/variable_hard_sphere_kernel.h>

#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>

namespace {

using atlas::Atom;
using atlas::Float3;
using atlas::Material;
using atlas::VariableHardSphereKernel;
using atlas::VariableSoftSphereKernel;

/** The leaf is a stateless POD, so it must be copyable into the device union and default-usable. */
static_assert(std::is_trivially_copyable_v<VariableSoftSphereKernel>);
static_assert(std::is_default_constructible_v<VariableSoftSphereKernel>);

/**
 * @brief Build an atomic species from its VHS/VSS collision parameters.
 * @param mass                  Species mass in kg.
 * @param reference_diameter    VHS/VSS reference diameter d_ref in m.
 * @param reference_temperature VHS/VSS reference temperature T_ref in K.
 * @param viscosity_index       VHS viscosity exponent omega.
 * @param scattering_parameter  VSS scattering exponent alpha (1 = isotropic).
 */
Material
species(const float mass,
        const float reference_diameter,
        const float reference_temperature,
        const float viscosity_index,
        const float scattering_parameter) {
    return Material(
        Atom(mass, 0.0f, 0.0f, 0.0f, reference_diameter, reference_temperature, viscosity_index, scattering_parameter));
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

/** Scatter the pair once from @p engine and return the cosine of the sampled deflection angle chi. */
float
deflection_cosine(const VariableSoftSphereKernel& kernel,
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

/**
 * @brief Mean deflection cosine over many fixed-geometry collisions at a given VSS alpha.
 *
 * The pre-collision velocities are held fixed and only the injected engine drives cos(chi),
 * so the pair mean equals @p alpha. The engine is re-seeded with @p seed for every call, so
 * different alphas consume the identical u1 stream; per sample this makes the ordering
 * `2 u1^2 - 1 <= 2 u1 - 1 <= 2 sqrt(u1) - 1` exact, hence the mean ordering exact too.
 */
double
mean_deflection_cosine(const float alpha, const int sample_count, const unsigned seed) {
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f, alpha);
    const Material rhs = species(3.0f, 3.0e-10f, 273.0f, 0.75f, alpha);

    const VariableSoftSphereKernel kernel {};

    // Fixed pre-collision geometry: the angular law is now a function of the engine alone.
    const Float3 lhs_velocity(215.0f, -60.0f, 40.0f);
    const Float3 rhs_velocity(-85.0f, 130.0f, -25.0f);

    atlas::default_random_engine engine(seed);

    double sum = 0.0;
    for (int i = 0; i < sample_count; ++i) {
        sum += deflection_cosine(kernel, lhs, rhs, lhs_velocity, rhs_velocity, engine);
    }
    return sum / sample_count;
}

}

TEST(VariableSoftSphereKernel, CrossSectionForwardsToVariableHardSphere) {
    // VSS and VHS differ only in scattering, never in the total cross section.
    const Material lhs = species(6.63e-26f, 4.17e-10f, 273.0f, 0.75f, 1.4f);
    const Material rhs = species(4.65e-26f, 4.17e-10f, 273.0f, 0.74f, 1.4f);

    for (const float relative_speed : { 1.0f, 500.0f, 5000.0f }) {
        EXPECT_FLOAT_EQ(VariableSoftSphereKernel::cross_section(lhs, rhs, relative_speed),
                        VariableHardSphereKernel::cross_section(lhs, rhs, relative_speed));
    }
}

TEST(VariableSoftSphereKernel, CrossSectionFollowsRelativeSpeedPowerLaw) {
    // Inherited from VHS: sigma scales as g^(1 - 2 omega).
    const float viscosity_index = 0.75f;
    const Material lhs = species(6.63e-26f, 4.17e-10f, 273.0f, viscosity_index, 1.4f);
    const Material rhs = species(6.63e-26f, 4.17e-10f, 273.0f, viscosity_index, 1.4f);

    const float slow_speed = 500.0f;
    const float fast_speed = 2000.0f;

    const float slow = VariableSoftSphereKernel::cross_section(lhs, rhs, slow_speed);
    const float fast = VariableSoftSphereKernel::cross_section(lhs, rhs, fast_speed);

    const float exponent       = 1.0f - 2.0f * viscosity_index;
    const float expected_ratio = std::pow(slow_speed / fast_speed, exponent);

    EXPECT_GT(slow, 0.0f);
    EXPECT_GT(fast, 0.0f);
    EXPECT_NEAR(slow / fast, expected_ratio, expected_ratio * 5.0e-3f);
}

TEST(VariableSoftSphereKernel, AlphaOneReducesToIsotropicVariableHardSphere) {
    // With alpha == 1 the soft-sphere angular law collapses to the isotropic VHS branch, so both
    // kernels must produce bit-for-bit identical post-collision velocities from the same input.
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f, 1.0f);
    const Material rhs = species(4.0f, 3.0e-10f, 273.0f, 0.75f, 1.0f);

    const VariableSoftSphereKernel vss {};
    const VariableHardSphereKernel vhs {};

    const Float3 lhs_start(120.0f, -55.0f, 33.0f);
    const Float3 rhs_start(-70.0f, 90.0f, -18.0f);

    // Same seed for both leaves so they consume the identical variate stream.
    constexpr unsigned seed = 20260710u;

    atlas::default_random_engine engine_vss(seed);
    Float3                       lhs_vss = lhs_start;
    Float3                       rhs_vss = rhs_start;
    vss(lhs_vss, rhs_vss, lhs, rhs, engine_vss);

    atlas::default_random_engine engine_vhs(seed);
    Float3                       lhs_vhs = lhs_start;
    Float3                       rhs_vhs = rhs_start;
    vhs(lhs_vhs, rhs_vhs, lhs, rhs, engine_vhs);

    EXPECT_FLOAT_EQ(lhs_vss.x, lhs_vhs.x);
    EXPECT_FLOAT_EQ(lhs_vss.y, lhs_vhs.y);
    EXPECT_FLOAT_EQ(lhs_vss.z, lhs_vhs.z);
    EXPECT_FLOAT_EQ(rhs_vss.x, rhs_vhs.x);
    EXPECT_FLOAT_EQ(rhs_vss.y, rhs_vhs.y);
    EXPECT_FLOAT_EQ(rhs_vss.z, rhs_vhs.z);
}

TEST(VariableSoftSphereKernel, AlphaBiasesScatteringInDocumentedDirection) {
    // cos(chi) = 2 u^(1/alpha) - 1, so E[cos chi] = (alpha - 1) / (alpha + 1): alpha > 1 biases
    // forward (positive mean), alpha < 1 biases backward (negative mean), alpha == 1 is unbiased.
    // Re-seeding to the same seed per alpha shares one u1 stream, making the ordering exact.
    constexpr int      sample_count = 40000;
    constexpr unsigned seed         = 20260710u;

    const double mean_backward = mean_deflection_cosine(0.5f, sample_count, seed);
    const double mean_isotropic = mean_deflection_cosine(1.0f, sample_count, seed);
    const double mean_forward   = mean_deflection_cosine(2.0f, sample_count, seed);

    // Strict ordering: sqrt(u) >= u >= u^2 per sample forces backward < isotropic < forward.
    EXPECT_LT(mean_backward, mean_isotropic);
    EXPECT_LT(mean_isotropic, mean_forward);

    // Isotropic mean is zero; the biased means match the analytic (alpha - 1) / (alpha + 1).
    EXPECT_NEAR(mean_isotropic, 0.0, 0.05);
    EXPECT_NEAR(mean_backward, (0.5 - 1.0) / (0.5 + 1.0), 0.05); // -1/3
    EXPECT_NEAR(mean_forward, (2.0 - 1.0) / (2.0 + 1.0), 0.05);  // +1/3

    EXPECT_LT(mean_backward, 0.0);
    EXPECT_GT(mean_forward, 0.0);
}

TEST(VariableSoftSphereKernel, ConservesMomentumAndEnergyForEqualMasses) {
    // A non-unit alpha exercises the anisotropic pow branch of the scatter.
    const Material lhs = species(2.0f, 3.0e-10f, 273.0f, 0.75f, 1.4f);
    const Material rhs = species(2.0f, 3.0e-10f, 273.0f, 0.75f, 1.4f);

    const VariableSoftSphereKernel kernel {};

    Float3 lhs_velocity(145.0f, -65.0f, 50.0f);
    Float3 rhs_velocity(-95.0f, 75.0f, -35.0f);

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

TEST(VariableSoftSphereKernel, ConservesMomentumAndEnergyForUnequalMasses) {
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f, 1.4f);
    const Material rhs = species(7.0f, 4.0e-10f, 273.0f, 0.75f, 1.4f);

    const VariableSoftSphereKernel kernel {};

    Float3 lhs_velocity(205.0f, -32.0f, 18.0f);
    Float3 rhs_velocity(-58.0f, 115.0f, -72.0f);

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

TEST(VariableSoftSphereKernel, PreservesRelativeSpeedAndCenterOfMass) {
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f, 1.4f);
    const Material rhs = species(4.0f, 3.0e-10f, 273.0f, 0.75f, 1.4f);

    const VariableSoftSphereKernel kernel {};

    Float3 lhs_velocity(165.0f, -85.0f, 48.0f);
    Float3 rhs_velocity(-75.0f, 118.0f, -22.0f);

    const float  speed_pre  = (lhs_velocity - rhs_velocity).length();
    const Float3 center_pre = momentum(lhs_velocity, lhs, rhs_velocity, rhs) / (lhs.mass() + rhs.mass());

    atlas::default_random_engine engine(33u);
    kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine);

    const float  speed_post  = (lhs_velocity - rhs_velocity).length();
    const Float3 center_post = momentum(lhs_velocity, lhs, rhs_velocity, rhs) / (lhs.mass() + rhs.mass());

    // The soft-sphere angular law still redirects the relative velocity without changing its length.
    EXPECT_NEAR(speed_post, speed_pre, 1.0e-3f * speed_pre);
    EXPECT_NEAR(center_post.x, center_pre.x, 1.0e-3f * std::abs(center_pre.x) + 1.0e-2f);
    EXPECT_NEAR(center_post.y, center_pre.y, 1.0e-3f * std::abs(center_pre.y) + 1.0e-2f);
    EXPECT_NEAR(center_post.z, center_pre.z, 1.0e-3f * std::abs(center_pre.z) + 1.0e-2f);
}

TEST(VariableSoftSphereKernel, ScatterIsDeterministicForIdenticalSeed) {
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f, 1.4f);
    const Material rhs = species(4.0f, 3.0e-10f, 273.0f, 0.75f, 1.4f);

    const VariableSoftSphereKernel kernel {};

    const Float3 lhs_start(28.0f, -48.0f, 68.0f);
    const Float3 rhs_start(-38.0f, 58.0f, -78.0f);

    // Reproducibility now flows from the seed, not the pair's geometry.
    constexpr unsigned seed = 76543u;

    atlas::default_random_engine engine_first(seed);
    Float3                       lhs_first = lhs_start;
    Float3                       rhs_first = rhs_start;
    kernel(lhs_first, rhs_first, lhs, rhs, engine_first);

    atlas::default_random_engine engine_second(seed);
    Float3                       lhs_second = lhs_start;
    Float3                       rhs_second = rhs_start;
    kernel(lhs_second, rhs_second, lhs, rhs, engine_second);

    EXPECT_FLOAT_EQ(lhs_first.x, lhs_second.x);
    EXPECT_FLOAT_EQ(lhs_first.y, lhs_second.y);
    EXPECT_FLOAT_EQ(lhs_first.z, lhs_second.z);
    EXPECT_FLOAT_EQ(rhs_first.x, rhs_second.x);
    EXPECT_FLOAT_EQ(rhs_first.y, rhs_second.y);
    EXPECT_FLOAT_EQ(rhs_first.z, rhs_second.z);
}

TEST(VariableSoftSphereKernel, SuccessfulScatterAdvancesEngineExactlyTwice) {
    // A resolved collision draws exactly two variates: one deflection cosine, one azimuth.
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f, 1.4f);
    const Material rhs = species(4.0f, 3.0e-10f, 273.0f, 0.75f, 1.4f);

    const VariableSoftSphereKernel kernel {};

    constexpr unsigned seed = 424242u;

    atlas::default_random_engine engine_a(seed);
    atlas::default_random_engine engine_b(seed);

    Float3 lhs_velocity(120.0f, -55.0f, 33.0f);
    Float3 rhs_velocity(-70.0f, 90.0f, -18.0f);
    kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine_a);

    // Manually skipping two draws must leave engine_b in the same state as the scattered engine_a.
    engine_b.discard(2);

    EXPECT_EQ(engine_a(), engine_b());
}

TEST(VariableSoftSphereKernel, DegeneratePairConsumesNoVariates) {
    // A zero relative speed is a no-op scatter, so the engine must be left untouched.
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f, 1.4f);
    const Material rhs = species(4.0f, 3.0e-10f, 273.0f, 0.75f, 1.4f);

    const VariableSoftSphereKernel kernel {};

    constexpr unsigned seed = 918273u;

    atlas::default_random_engine engine_a(seed);
    atlas::default_random_engine engine_b(seed);

    // Identical velocities give zero relative speed, hitting the degenerate early return.
    Float3 lhs_velocity(64.0f, -12.0f, 27.0f);
    Float3 rhs_velocity(64.0f, -12.0f, 27.0f);
    kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine_a);

    EXPECT_EQ(engine_a(), engine_b());
}
