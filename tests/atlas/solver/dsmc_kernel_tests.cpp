#include <atlas/solver/dsmc/kernel/dsmc_kernel.h>

#include <atlas/material/atom.h>
#include <atlas/material/material.h>
#include <atlas/material/molecule.h>
#include <atlas/math/math.h>
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
