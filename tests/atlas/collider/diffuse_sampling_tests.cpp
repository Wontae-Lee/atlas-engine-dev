#include <atlas/collider/diffuse_sampling.h>

#include <atlas/collider/isothermal_collider.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/math/math.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using atlas::DiffuseSampling;
using atlas::Geometry;
using atlas::IsothermalCollider;
using atlas::Plane;
using atlas::Quaternion;
using atlas::Sync;
using atlas::SyncHostPtr;
using atlas::Unit;
using atlas::Float3;
using atlas::tol;

bool
is_finite_vec(const Float3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

SyncHostPtr
make_identity_sync() {
    return Sync::builder()
        .with_rigid_pose(Float3(0.0f, 0.0f, 0.0f), Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
        .make_host_shared();
}

Unit
make_static_unit() {
    return Unit::builder()
        .with_geometry(Geometry(Plane(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 1.0f))))
        .with_sync(make_identity_sync())
        .build();
}

// Fully diffuse collider (momentum_acc == 1) so reflect() always takes the
// hemisphere-sampling branch selected by `sampling`.
IsothermalCollider
make_diffuse_collider(const DiffuseSampling sampling, const float restitution = 1.0f) {
    return IsothermalCollider::builder()
        .with_unit(make_static_unit())
        .with_momentum_accommodation_coefficient(1.0f)
        .with_restitution(restitution)
        .with_diffuse_sampling(sampling)
        .build();
}

}

TEST(DiffuseSampling, ModesAreDistinct) {
    EXPECT_NE(DiffuseSampling::cosine_weighted, DiffuseSampling::uniform);
}

TEST(DiffuseSampling, CosineReflectIsFinite) {
    const auto collider = make_diffuse_collider(DiffuseSampling::cosine_weighted);

    const Float3 reflected = collider.reflect(Float3(0.3f, -0.4f, -1.0f), Float3(0.0f, 0.0f, 1.0f));

    EXPECT_TRUE(is_finite_vec(reflected));
}

TEST(DiffuseSampling, UniformReflectIsFinite) {
    const auto collider = make_diffuse_collider(DiffuseSampling::uniform);

    const Float3 reflected = collider.reflect(Float3(0.3f, -0.4f, -1.0f), Float3(0.0f, 0.0f, 1.0f));

    EXPECT_TRUE(is_finite_vec(reflected));
}

TEST(DiffuseSampling, CosineReflectConservesSpeed) {
    const auto   collider = make_diffuse_collider(DiffuseSampling::cosine_weighted);
    const Float3 incident(1.0f, 2.0f, -3.0f);

    // The outgoing direction is a unit vector, so speed is preserved regardless
    // of whether the specular or the diffuse branch wins the coin flip.
    const Float3 reflected = collider.reflect(incident, Float3(0.0f, 0.0f, 1.0f));

    EXPECT_NEAR(reflected.length(), incident.length(), tol);
}

TEST(DiffuseSampling, UniformReflectConservesSpeed) {
    const auto   collider = make_diffuse_collider(DiffuseSampling::uniform);
    const Float3 incident(1.0f, 2.0f, -3.0f);

    const Float3 reflected = collider.reflect(incident, Float3(0.0f, 0.0f, 1.0f));

    EXPECT_NEAR(reflected.length(), incident.length(), tol);
}

TEST(DiffuseSampling, ReflectScalesByRestitution) {
    const auto   collider = make_diffuse_collider(DiffuseSampling::uniform, 0.25f);
    const Float3 incident(0.0f, 0.0f, -2.0f);

    const Float3 reflected = collider.reflect(incident, Float3(0.0f, 0.0f, 1.0f));

    EXPECT_NEAR(reflected.length(), incident.length() * 0.25f, tol);
}

TEST(DiffuseSampling, CosineReflectStaysInHemisphere) {
    const auto   collider = make_diffuse_collider(DiffuseSampling::cosine_weighted);
    const Float3 normal(0.0f, 0.0f, 1.0f);

    // Every incident heads into the +z wall; both branches must emit back into
    // the hemisphere about the normal (non-negative projection).
    const Float3 incidents[] = {
        Float3(0.0f, 0.0f, -1.0f),
        Float3(1.0f, 0.0f, -1.0f),
        Float3(-0.5f, 0.7f, -1.3f),
        Float3(2.0f, -1.0f, -0.5f),
    };

    for (const Float3& incident : incidents) {
        const Float3 reflected = collider.reflect(incident, normal);
        EXPECT_GE(reflected.dot(normal), -tol);
    }
}

TEST(DiffuseSampling, UniformReflectStaysInHemisphere) {
    const auto   collider = make_diffuse_collider(DiffuseSampling::uniform);
    const Float3 normal(0.0f, 0.0f, 1.0f);

    const Float3 incidents[] = {
        Float3(0.0f, 0.0f, -1.0f),
        Float3(1.0f, 0.0f, -1.0f),
        Float3(-0.5f, 0.7f, -1.3f),
        Float3(2.0f, -1.0f, -0.5f),
    };

    for (const Float3& incident : incidents) {
        const Float3 reflected = collider.reflect(incident, normal);
        EXPECT_GE(reflected.dot(normal), -tol);
    }
}

TEST(DiffuseSampling, ReflectIsDeterministic) {
    // The diffuse variates are hashed from the incident/normal geometry, so the
    // same inputs must reproduce the same outgoing velocity with no RNG state.
    const auto   collider = make_diffuse_collider(DiffuseSampling::cosine_weighted);
    const Float3 incident(0.6f, -0.2f, -0.9f);
    const Float3 normal(0.0f, 0.0f, 1.0f);

    const Float3 first  = collider.reflect(incident, normal);
    const Float3 second = collider.reflect(incident, normal);

    EXPECT_FLOAT_EQ(first.x, second.x);
    EXPECT_FLOAT_EQ(first.y, second.y);
    EXPECT_FLOAT_EQ(first.z, second.z);
}
