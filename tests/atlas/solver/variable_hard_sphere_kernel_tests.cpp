#include <atlas/solver/dsmc/kernel/variable_hard_sphere_kernel.h>

#include <atlas/material/atom.h>
#include <atlas/material/material.h>
#include <atlas/material/solid.h>
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
using atlas::Material;
using atlas::Solid;
using atlas::VariableHardSphereKernel;

/** The leaf is a stateless POD, so it must be copyable into the device union and default-usable. */
static_assert(std::is_trivially_copyable_v<VariableHardSphereKernel>);
static_assert(std::is_default_constructible_v<VariableHardSphereKernel>);

/**
 * @brief Build an atomic species from its VHS collision parameters.
 * @param mass                Species mass in kg.
 * @param reference_diameter  VHS reference diameter d_ref in m.
 * @param reference_temperature VHS reference temperature T_ref in K.
 * @param viscosity_index     VHS viscosity exponent omega.
 */
Material
species(const float mass,
        const float reference_diameter,
        const float reference_temperature,
        const float viscosity_index) {
    return Material(Atom(mass, 0.0f, 0.0f, 0.0f, reference_diameter, reference_temperature, viscosity_index, 1.0f));
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
 * @brief Scatter the pair once, drawing from @p engine, and return the cosine of the sampled
 *        deflection angle chi. The velocities are taken by value so one fixed geometry can be
 *        re-scattered repeatedly against a single advancing engine.
 */
float
deflection_cosine(const VariableHardSphereKernel& kernel,
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

TEST(VariableHardSphereKernel, CrossSectionMatchesAnalyticFormula) {
    // Well-conditioned realistic parameters keep every float intermediate finite, so the kernel's
    // single-precision result should track an independent double-precision evaluation of the
    // documented VHS law sigma = pi d_ref^2 (2 k T_ref / (m_r g^2))^(omega - 1/2) / Gamma(5/2 - omega).
    const float mass                  = 6.63e-26f;
    const float reference_diameter    = 4.17e-10f;
    const float reference_temperature = 273.0f;
    const float viscosity_index       = 0.75f;
    const float relative_speed        = 1000.0f;

    const Material lhs = species(mass, reference_diameter, reference_temperature, viscosity_index);
    const Material rhs = species(mass, reference_diameter, reference_temperature, viscosity_index);

    const double reduced_mass   = static_cast<double>(mass) * mass / (static_cast<double>(mass) + mass);
    const double thermal_ratio  = (2.0 * static_cast<double>(atlas::boltzmann_constant) * reference_temperature)
        / (reduced_mass * static_cast<double>(relative_speed) * relative_speed);
    const double expected = static_cast<double>(atlas::pi) * reference_diameter * reference_diameter
        * std::pow(thermal_ratio, static_cast<double>(viscosity_index) - 0.5)
        / std::tgamma(2.5 - static_cast<double>(viscosity_index));

    const float cross_section = VariableHardSphereKernel::cross_section(lhs, rhs, relative_speed);

    EXPECT_NEAR(cross_section, static_cast<float>(expected), static_cast<float>(expected) * 2.0e-2f);
}

TEST(VariableHardSphereKernel, CrossSectionFollowsRelativeSpeedPowerLaw) {
    // sigma scales as g^(1 - 2 omega): the ratio at two speeds must equal (g1 / g2)^(1 - 2 omega),
    // with every speed-independent factor (area, Gamma, reduced mass) cancelling out.
    const float viscosity_index = 0.75f;
    const Material lhs = species(6.63e-26f, 4.17e-10f, 273.0f, viscosity_index);
    const Material rhs = species(6.63e-26f, 4.17e-10f, 273.0f, viscosity_index);

    const float slow_speed = 500.0f;
    const float fast_speed = 2000.0f;

    const float slow = VariableHardSphereKernel::cross_section(lhs, rhs, slow_speed);
    const float fast = VariableHardSphereKernel::cross_section(lhs, rhs, fast_speed);

    const float exponent       = 1.0f - 2.0f * viscosity_index;
    const float expected_ratio = std::pow(slow_speed / fast_speed, exponent);

    EXPECT_GT(slow, 0.0f);
    EXPECT_GT(fast, 0.0f);
    EXPECT_NEAR(slow / fast, expected_ratio, expected_ratio * 5.0e-3f);
}

TEST(VariableHardSphereKernel, CrossSectionVanishesAtZeroRelativeSpeed) {
    const Material material = species(6.63e-26f, 4.17e-10f, 273.0f, 0.75f);

    EXPECT_FLOAT_EQ(VariableHardSphereKernel::cross_section(material, material, 0.0f), 0.0f);
}

TEST(VariableHardSphereKernel, CrossSectionIsZeroForMasslessSpecies) {
    const Material massless = species(0.0f, 4.17e-10f, 273.0f, 0.75f);
    const Material normal   = species(6.63e-26f, 4.17e-10f, 273.0f, 0.75f);

    EXPECT_FLOAT_EQ(VariableHardSphereKernel::cross_section(massless, normal, 1000.0f), 0.0f);
}

TEST(VariableHardSphereKernel, CrossSectionIsZeroForOutOfRangeViscosityIndex) {
    // A viscosity index at or above 5/2 drives the Gamma argument (5/2 - omega) non-positive,
    // which the kernel rejects rather than returning a NaN.
    const Material material = species(6.63e-26f, 4.17e-10f, 273.0f, 3.0f);

    EXPECT_FLOAT_EQ(VariableHardSphereKernel::cross_section(material, material, 1000.0f), 0.0f);
}

TEST(VariableHardSphereKernel, SolidPartnerStaysFiniteViaUnitStubs) {
    // Solid stubs every non-mass property to 1.0f precisely so the VHS math (which divides by and
    // takes powers of reference diameter/temperature and the viscosity index) stays well-defined
    // rather than producing NaN/Inf when a wall species is accidentally fed to the kernel. With
    // mass > 0 and every stub positive, the cross section must be finite and positive.
    const Material solid  = Material(Solid(5.0e-26f));
    const Material normal = species(6.63e-26f, 4.17e-10f, 273.0f, 0.75f);

    for (const float relative_speed : { 1.0f, 500.0f, 5000.0f }) {
        const float solid_pair = VariableHardSphereKernel::cross_section(solid, solid, relative_speed);
        const float mixed_pair = VariableHardSphereKernel::cross_section(solid, normal, relative_speed);

        EXPECT_TRUE(std::isfinite(solid_pair)) << "relative_speed = " << relative_speed;
        EXPECT_GT(solid_pair, 0.0f) << "relative_speed = " << relative_speed;
        EXPECT_TRUE(std::isfinite(mixed_pair)) << "relative_speed = " << relative_speed;
        EXPECT_GT(mixed_pair, 0.0f) << "relative_speed = " << relative_speed;
    }
}

TEST(VariableHardSphereKernel, ConservesMomentumAndEnergyForEqualMasses) {
    const Material lhs = species(2.0f, 3.0e-10f, 273.0f, 0.75f);
    const Material rhs = species(2.0f, 3.0e-10f, 273.0f, 0.75f);

    const VariableHardSphereKernel kernel {};

    Float3 lhs_velocity(160.0f, -70.0f, 40.0f);
    Float3 rhs_velocity(-100.0f, 85.0f, -30.0f);

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

TEST(VariableHardSphereKernel, ConservesMomentumAndEnergyForUnequalMasses) {
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f);
    const Material rhs = species(7.0f, 4.0e-10f, 273.0f, 0.75f);

    const VariableHardSphereKernel kernel {};

    Float3 lhs_velocity(210.0f, -35.0f, 20.0f);
    Float3 rhs_velocity(-55.0f, 120.0f, -75.0f);

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

TEST(VariableHardSphereKernel, PreservesRelativeSpeedAndCenterOfMass) {
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f);
    const Material rhs = species(4.0f, 3.0e-10f, 273.0f, 0.75f);

    const VariableHardSphereKernel kernel {};

    Float3 lhs_velocity(175.0f, -95.0f, 55.0f);
    Float3 rhs_velocity(-65.0f, 125.0f, -25.0f);

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

TEST(VariableHardSphereKernel, ScatterIsIsotropic) {
    // VHS shares the isotropic angular law with hard spheres: cos(chi) is uniform on [-1, 1].
    // Hold ONE fixed pre-collision geometry and re-scatter it from a single seeded engine, so the
    // distribution is drawn purely from the RNG. This exposes any velocity-dependent bias the old
    // geometry-hash test could not. The seed keeps the histogram deterministic, not flaky.
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f);
    const Material rhs = species(3.0f, 3.0e-10f, 273.0f, 0.75f);

    const VariableHardSphereKernel kernel {};

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

    EXPECT_NEAR(cosine_sum / sample_count, 0.0, 0.05);
}

TEST(VariableHardSphereKernel, ScatterIsReproducibleForIdenticalSeeds) {
    // Reproducibility now flows from the engine seed, not the pair geometry: identical input plus
    // two identically seeded engines must scatter to bit-for-bit identical states.
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f);
    const Material rhs = species(4.0f, 3.0e-10f, 273.0f, 0.75f);

    const VariableHardSphereKernel kernel {};

    const Float3 lhs_start(25.0f, -45.0f, 65.0f);
    const Float3 rhs_start(-35.0f, 55.0f, -75.0f);

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

TEST(VariableHardSphereKernel, ScatterDiffersForDifferentSeeds) {
    // With the geometry hash gone, differently seeded engines draw different angles, so the same
    // input pair reaches a different post-collision state.
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f);
    const Material rhs = species(4.0f, 3.0e-10f, 273.0f, 0.75f);

    const VariableHardSphereKernel kernel {};

    const Float3 lhs_start(25.0f, -45.0f, 65.0f);
    const Float3 rhs_start(-35.0f, 55.0f, -75.0f);

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

TEST(VariableHardSphereKernel, DegeneratePairDoesNotConsumeVariates) {
    // A zero-relative-speed pair takes the early return, leaving the engine untouched. The first
    // engine is never advanced, so its next draw equals a twin engine's first draw.
    const Material lhs = species(2.0f, 3.0e-10f, 273.0f, 0.75f);
    const Material rhs = species(2.0f, 3.0e-10f, 273.0f, 0.75f);

    const VariableHardSphereKernel kernel {};

    // Equal velocities -> zero relative speed -> degenerate scatter.
    Float3 lhs_velocity(50.0f, -20.0f, 10.0f);
    Float3 rhs_velocity(50.0f, -20.0f, 10.0f);

    atlas::default_random_engine engine_a(13579u);
    atlas::default_random_engine engine_b(13579u);

    kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine_a);

    EXPECT_EQ(engine_a(), engine_b());
}

TEST(VariableHardSphereKernel, SuccessfulScatterAdvancesEngineExactlyTwice) {
    // A successful scatter draws exactly two variates (deflection cosine, azimuth). Discarding two
    // from a twin engine leaves both in the same state.
    const Material lhs = species(1.0f, 3.0e-10f, 273.0f, 0.75f);
    const Material rhs = species(4.0f, 3.0e-10f, 273.0f, 0.75f);

    const VariableHardSphereKernel kernel {};

    Float3 lhs_velocity(120.0f, -40.0f, 25.0f);
    Float3 rhs_velocity(-60.0f, 80.0f, -15.0f);

    atlas::default_random_engine engine_a(24680u);
    atlas::default_random_engine engine_b(24680u);

    kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine_a);
    engine_b.discard(2);

    EXPECT_EQ(engine_a(), engine_b());
}
