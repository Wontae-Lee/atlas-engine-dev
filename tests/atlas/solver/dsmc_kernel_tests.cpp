#include <atlas/solver/dsmc/kernel/dsmc_kernel.h>

#include <atlas/material/atom.h>
#include <atlas/material/material.h>
#include <atlas/material/molecule.h>
#include <atlas/math/math.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/solver/dsmc/kernel/hard_sphere_kernel.h>
#include <atlas/solver/dsmc/kernel/variable_hard_sphere_kernel.h>
#include <atlas/solver/dsmc/kernel/variable_soft_sphere_kernel.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using atlas::Atom;
using atlas::DsmcKernel;
using atlas::DsmcKernelType;
using atlas::HardSphereKernel;
using atlas::Material;
using atlas::Molecule;
using atlas::VariableHardSphereKernel;
using atlas::VariableSoftSphereKernel;

// Nitrogen: a mass of 4.65e-26 kg squares to 2.16e-51, far below the smallest
// float subnormal. A reduced mass computed as (m * m) / (m + m) flushes to zero.
Material
nitrogen() {
    return Material(Molecule(4.65e-26f, 0.0f, 0.0f, 0.0f, 4.17e-10f, 273.0f, 0.74f, 1.0f));
}

Material
argon() {
    return Material(Atom(6.63e-26f, 0.0f, 0.0f, 0.0f, 4.17e-10f, 273.0f, 0.81f, 1.0f));
}

}

TEST(VariableHardSphereKernel, CrossSectionIsFiniteForMolecularMasses) {
    const Material lhs = nitrogen();
    const Material rhs = nitrogen();

    for (const float relative_speed : { 1.0e-9f, 1.0e-3f, 1.0f, 100.0f, 1000.0f, 10000.0f }) {
        const float cross_section = VariableHardSphereKernel::cross_section(lhs, rhs, relative_speed);

        EXPECT_TRUE(std::isfinite(cross_section)) << "relative_speed = " << relative_speed;
        EXPECT_GE(cross_section, 0.0f) << "relative_speed = " << relative_speed;
    }
}

TEST(VariableHardSphereKernel, CrossSectionIsPositiveAtThermalSpeed) {
    const float cross_section = VariableHardSphereKernel::cross_section(nitrogen(), nitrogen(), 1000.0f);

    EXPECT_TRUE(std::isfinite(cross_section));
    EXPECT_GT(cross_section, 0.0f);

    // The VHS cross-section falls below the reference area for a relative speed
    // above the reference thermal speed.
    const float reference_area = atlas::pi * 4.17e-10f * 4.17e-10f;

    EXPECT_LT(cross_section, reference_area);
}

TEST(VariableHardSphereKernel, CrossSectionIsFiniteForMixedSpecies) {
    const float cross_section = VariableHardSphereKernel::cross_section(nitrogen(), argon(), 800.0f);

    EXPECT_TRUE(std::isfinite(cross_section));
    EXPECT_GT(cross_section, 0.0f);
}

TEST(VariableHardSphereKernel, CrossSectionIsZeroAtZeroRelativeSpeed) {
    EXPECT_FLOAT_EQ(VariableHardSphereKernel::cross_section(nitrogen(), nitrogen(), 0.0f), 0.0f);
}

TEST(VariableHardSphereKernel, SigmaTimesGVanishesAsRelativeSpeedVanishes) {
    const Material lhs = nitrogen();
    const Material rhs = nitrogen();

    const float fast = VariableHardSphereKernel::cross_section(lhs, rhs, 1000.0f) * 1000.0f;
    const float slow = VariableHardSphereKernel::cross_section(lhs, rhs, 1.0e-3f) * 1.0e-3f;

    EXPECT_TRUE(std::isfinite(fast));
    EXPECT_TRUE(std::isfinite(slow));
    EXPECT_LT(slow, fast);
}

TEST(VariableSoftSphereKernel, CrossSectionMatchesVariableHardSphere) {
    const Material lhs = nitrogen();
    const Material rhs = argon();

    for (const float relative_speed : { 1.0f, 500.0f, 5000.0f }) {
        EXPECT_FLOAT_EQ(VariableSoftSphereKernel::cross_section(lhs, rhs, relative_speed),
                        VariableHardSphereKernel::cross_section(lhs, rhs, relative_speed));
    }
}

TEST(HardSphereKernel, CrossSectionIsTheReferenceArea) {
    const float cross_section = HardSphereKernel::cross_section(nitrogen(), nitrogen(), 1000.0f);
    const float reference_area = atlas::pi * 4.17e-10f * 4.17e-10f;

    EXPECT_TRUE(std::isfinite(cross_section));
    EXPECT_NEAR(cross_section, reference_area, reference_area * 1.0e-5f);
}

TEST(DsmcKernel, SigmaGIsFiniteForEveryKernelType) {
    const Material materials[2] = { nitrogen(), argon() };

    for (const DsmcKernelType type : { DsmcKernelType::hard_sphere,
                                       DsmcKernelType::variable_hard_sphere,
                                       DsmcKernelType::variable_soft_sphere }) {
        const DsmcKernel kernel(type);

        for (const float relative_speed : { 1.0e-6f, 1.0f, 1000.0f }) {
            const float sigma_g = kernel.sigma_g(materials, 0, 1, relative_speed * relative_speed);

            EXPECT_TRUE(std::isfinite(sigma_g))
                << "type = " << static_cast<int>(type) << ", relative_speed = " << relative_speed;
            EXPECT_GE(sigma_g, 0.0f);
        }
    }
}

TEST(DsmcKernel, DefaultConstructsToHardSphereLeaf) {
    EXPECT_EQ(DsmcKernel().type, DsmcKernelType::hard_sphere);
}

TEST(DsmcKernel, ExplicitConstructorActivatesRequestedLeaf) {
    for (const DsmcKernelType type : { DsmcKernelType::hard_sphere,
                                       DsmcKernelType::variable_hard_sphere,
                                       DsmcKernelType::variable_soft_sphere }) {
        EXPECT_EQ(DsmcKernel(type).type, type);
    }
}

TEST(DsmcKernel, CrossSectionDispatchesToActiveLeaf) {
    const Material lhs = nitrogen();
    const Material rhs = argon();
    const float    relative_speed = 800.0f;

    EXPECT_FLOAT_EQ(DsmcKernel(DsmcKernelType::hard_sphere).cross_section(lhs, rhs, relative_speed),
                    HardSphereKernel::cross_section(lhs, rhs, relative_speed));
    EXPECT_FLOAT_EQ(DsmcKernel(DsmcKernelType::variable_hard_sphere).cross_section(lhs, rhs, relative_speed),
                    VariableHardSphereKernel::cross_section(lhs, rhs, relative_speed));
    EXPECT_FLOAT_EQ(DsmcKernel(DsmcKernelType::variable_soft_sphere).cross_section(lhs, rhs, relative_speed),
                    VariableSoftSphereKernel::cross_section(lhs, rhs, relative_speed));
}

TEST(DsmcKernel, CopyConstructorPreservesActiveLeaf) {
    const DsmcKernel original(DsmcKernelType::variable_soft_sphere);
    const DsmcKernel copy(original);

    EXPECT_EQ(copy.type, DsmcKernelType::variable_soft_sphere);
    EXPECT_FLOAT_EQ(copy.cross_section(nitrogen(), argon(), 500.0f),
                    original.cross_section(nitrogen(), argon(), 500.0f));
}

TEST(DsmcKernel, CopyAssignmentPreservesActiveLeaf) {
    const DsmcKernel source(DsmcKernelType::variable_hard_sphere);
    DsmcKernel       target(DsmcKernelType::hard_sphere);

    target = source;

    EXPECT_EQ(target.type, DsmcKernelType::variable_hard_sphere);
    EXPECT_FLOAT_EQ(target.cross_section(nitrogen(), nitrogen(), 700.0f),
                    VariableHardSphereKernel::cross_section(nitrogen(), nitrogen(), 700.0f));
}

TEST(DsmcKernel, SigmaGIsZeroForNonPositiveRelativeSpeedSquared) {
    const Material   materials[2] = { nitrogen(), argon() };
    const DsmcKernel kernel(DsmcKernelType::variable_hard_sphere);

    EXPECT_FLOAT_EQ(kernel.sigma_g(materials, 0, 1, 0.0f), 0.0f);
    EXPECT_FLOAT_EQ(kernel.sigma_g(materials, 0, 1, -4.0f), 0.0f);
}

TEST(DsmcKernel, ScatterThroughUmbrellaConservesMomentum) {
    // Unit-scale masses keep the momentum arithmetic well conditioned; the leaf still splits
    // the post-collision velocities by the mass ratio.
    const Material lhs = Material(Atom(1.0f, 0.0f, 0.0f, 0.0f, 1.0e-10f, 273.0f, 0.5f, 1.0f));
    const Material rhs = Material(Atom(3.0f, 0.0f, 0.0f, 0.0f, 1.0e-10f, 273.0f, 0.5f, 1.0f));

    const DsmcKernel kernel(DsmcKernelType::hard_sphere);

    atlas::Float3 lhs_velocity(120.0f, -40.0f, 25.0f);
    atlas::Float3 rhs_velocity(-60.0f, 80.0f, -15.0f);

    const atlas::Float3 momentum_pre = lhs_velocity * lhs.mass() + rhs_velocity * rhs.mass();

    atlas::default_random_engine engine(1234u);
    kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine);

    const atlas::Float3 momentum_post = lhs_velocity * lhs.mass() + rhs_velocity * rhs.mass();

    EXPECT_TRUE(atlas::isfinite(lhs_velocity));
    EXPECT_TRUE(atlas::isfinite(rhs_velocity));
    EXPECT_NEAR(momentum_post.x, momentum_pre.x, 1.0e-3f * std::abs(momentum_pre.x) + 1.0e-2f);
    EXPECT_NEAR(momentum_post.y, momentum_pre.y, 1.0e-3f * std::abs(momentum_pre.y) + 1.0e-2f);
    EXPECT_NEAR(momentum_post.z, momentum_pre.z, 1.0e-3f * std::abs(momentum_pre.z) + 1.0e-2f);
}

TEST(DsmcKernel, SigmaGEqualsCrossSectionTimesRelativeSpeed) {
    // sigma_g takes the SQUARED relative speed and must internally take the square root once,
    // returning cross_section(pair, g) * g. Feeding g*g here and comparing against an independent
    // cross_section(..., g) * g nails that contract for every leaf.
    const Material materials[2] = { nitrogen(), argon() };
    const float    relative_speed = 1000.0f;

    for (const DsmcKernelType type : { DsmcKernelType::hard_sphere,
                                       DsmcKernelType::variable_hard_sphere,
                                       DsmcKernelType::variable_soft_sphere }) {
        const DsmcKernel kernel(type);

        const float expected = kernel.cross_section(materials[0], materials[1], relative_speed) * relative_speed;
        const float actual   = kernel.sigma_g(materials, 0, 1, relative_speed * relative_speed);

        EXPECT_GT(expected, 0.0f) << "type = " << static_cast<int>(type);
        EXPECT_NEAR(actual, expected, std::abs(expected) * 1.0e-4f)
            << "type = " << static_cast<int>(type);
    }
}

TEST(DsmcKernel, SigmaGIndexesMaterialsBySpecies) {
    // sigma_g reads materials[lhs_species] and materials[rhs_species] without a bounds check, so
    // it must actually honour the indices. Species 0 is a zero-diameter atom (hard-sphere cross
    // section 0) and species 1 a real atom, which lets the selected entries drive the outcome.
    const Material materials[2] = {
        Material(Atom(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 273.0f, 0.5f, 1.0f)),
        Material(Atom(1.0f, 0.0f, 0.0f, 0.0f, 3.0e-10f, 273.0f, 0.5f, 1.0f)),
    };

    const DsmcKernel kernel(DsmcKernelType::hard_sphere);
    const float      relative_speed_squared = 500.0f * 500.0f;

    // A pair of zero-diameter species carries no cross section; a pair of real ones does.
    EXPECT_FLOAT_EQ(kernel.sigma_g(materials, 0, 0, relative_speed_squared), 0.0f);
    EXPECT_GT(kernel.sigma_g(materials, 1, 1, relative_speed_squared), 0.0f);

    // The cross section is symmetric in the pair, so swapping the indices must not change it.
    EXPECT_FLOAT_EQ(kernel.sigma_g(materials, 0, 1, relative_speed_squared),
                    kernel.sigma_g(materials, 1, 0, relative_speed_squared));
}

TEST(DsmcKernel, NormalizesOutOfRangeTypeToHardSphere) {
    // The explicit constructor routes through DeviceVariant::normalize, so an out-of-range tag
    // folds to the default hard-sphere arm rather than selecting a non-existent union member.
    const DsmcKernel kernel(static_cast<DsmcKernelType>(999));

    EXPECT_EQ(kernel.type, DsmcKernelType::hard_sphere);
    EXPECT_FLOAT_EQ(kernel.cross_section(nitrogen(), argon(), 800.0f),
                    HardSphereKernel::cross_section(nitrogen(), argon(), 800.0f));
}

TEST(DsmcKernel, ScatterForwardsEngineToActiveLeaf) {
    // The umbrella must draw from the same stream and produce the same post-collision state as
    // invoking the active leaf directly with an identically seeded engine.
    const Material lhs = Material(Atom(1.0f, 0.0f, 0.0f, 0.0f, 1.0e-10f, 273.0f, 0.5f, 1.0f));
    const Material rhs = Material(Atom(3.0f, 0.0f, 0.0f, 0.0f, 1.0e-10f, 273.0f, 0.5f, 1.0f));

    const DsmcKernel       umbrella(DsmcKernelType::hard_sphere);
    const HardSphereKernel leaf {};

    constexpr unsigned  seed = 13572468u;
    const atlas::Float3 lhs_start(120.0f, -40.0f, 25.0f);
    const atlas::Float3 rhs_start(-60.0f, 80.0f, -15.0f);

    atlas::default_random_engine engine_umbrella(seed);
    atlas::Float3                lhs_umbrella = lhs_start;
    atlas::Float3                rhs_umbrella = rhs_start;
    umbrella(lhs_umbrella, rhs_umbrella, lhs, rhs, engine_umbrella);

    atlas::default_random_engine engine_leaf(seed);
    atlas::Float3                lhs_leaf = lhs_start;
    atlas::Float3                rhs_leaf = rhs_start;
    leaf(lhs_leaf, rhs_leaf, lhs, rhs, engine_leaf);

    EXPECT_FLOAT_EQ(lhs_umbrella.x, lhs_leaf.x);
    EXPECT_FLOAT_EQ(lhs_umbrella.y, lhs_leaf.y);
    EXPECT_FLOAT_EQ(lhs_umbrella.z, lhs_leaf.z);
    EXPECT_FLOAT_EQ(rhs_umbrella.x, rhs_leaf.x);
    EXPECT_FLOAT_EQ(rhs_umbrella.y, rhs_leaf.y);
    EXPECT_FLOAT_EQ(rhs_umbrella.z, rhs_leaf.z);
}
