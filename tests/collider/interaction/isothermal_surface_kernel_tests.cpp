#include <atlas/collider/interaction/isothermal_surface_kernel.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using atlas::DiffuseSampling;
using atlas::IsothermalSurfaceInteraction;
using atlas::Float3;
using atlas::tol;

bool
is_finite_vec(const Float3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

}

TEST(IsothermalSurfaceKernel, DiffuseSamplingModesAreDistinct) {
    EXPECT_NE(DiffuseSampling::cosine_weighted, DiffuseSampling::uniform);
}

TEST(IsothermalSurfaceKernel, HeaderExportsUsableInteractionKernel) {
    const auto kernel = IsothermalSurfaceInteraction::builder()
                            .with_diffuse_sampling(DiffuseSampling::uniform)
                            .with_restitution(0.25f)
                            .with_momentum_acc(0.0f)
                            .build();

    const Float3 incident(0.0f, -4.0f, 0.0f);
    const Float3 out = kernel(incident, Float3(0.0f, 1.0f, 0.0f));

    EXPECT_TRUE(is_finite_vec(out));
    EXPECT_NEAR(out.x, 0.0f, tol);
    EXPECT_NEAR(out.y, 1.0f, tol);
    EXPECT_NEAR(out.z, 0.0f, tol);
}
