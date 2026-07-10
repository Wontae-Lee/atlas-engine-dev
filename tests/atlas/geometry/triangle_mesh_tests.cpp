#include <atlas/geometry/triangle_mesh.h>

// triangle_mesh.h only forward-declares Geometry, which make_device_geometry_view() returns
// by value and the trace tests below call through.
#include <atlas/geometry/geometry.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace {

using atlas::AABB;
using atlas::Float3;
using atlas::Geometry;
using atlas::GeometryType;
using atlas::HitSurface;
using atlas::HostBuffer;
using atlas::Ray;
using atlas::TriangleContainer4;
using atlas::TriangleMesh;
using atlas::TriangleMeshView;

constexpr float kTol = 1e-4f;

/**
 * The eight corners of the unit cube [0,1]^3, shared by the owning-mesh and the
 * view test fixtures.
 */
const std::vector<Float3>&
cube_vertices() {
    static const std::vector<Float3> verts = {
        Float3(0.0f, 0.0f, 0.0f), // 0
        Float3(1.0f, 0.0f, 0.0f), // 1
        Float3(1.0f, 1.0f, 0.0f), // 2
        Float3(0.0f, 1.0f, 0.0f), // 3
        Float3(0.0f, 0.0f, 1.0f), // 4
        Float3(1.0f, 0.0f, 1.0f), // 5
        Float3(1.0f, 1.0f, 1.0f), // 6
        Float3(0.0f, 1.0f, 1.0f), // 7
    };
    return verts;
}

/**
 * Twelve triangles (two per face) triangulating the unit cube with a consistent
 * outward winding, so the mesh is a closed, oriented surface suitable for the
 * winding-number inside test.
 */
const std::vector<int>&
cube_indices() {
    static const std::vector<int> idx = {
        0, 3, 2, 0, 2, 1, // -Z (bottom)
        4, 5, 6, 4, 6, 7, // +Z (top)
        0, 1, 5, 0, 5, 4, // -Y
        3, 7, 6, 3, 6, 2, // +Y
        0, 4, 7, 0, 7, 3, // -X
        1, 2, 6, 1, 6, 5, // +X
    };
    return idx;
}

/** Packs three vertices plus their derived geometric normal into a container. */
TriangleContainer4
make_tri(const Float3& a, const Float3& b, const Float3& c) {
    const Float3 n = atlas::normalized_or(
        atlas::cross(b - a, c - a),
        Float3(0.0f, 0.0f, 1.0f));
    return TriangleContainer4(a, b, c, n);
}

/** Builds the cube as a triangle buffer for constructing a TriangleMesh. */
HostBuffer<TriangleContainer4>
cube_triangles() {
    const std::vector<Float3>& v = cube_vertices();
    const std::vector<int>& idx  = cube_indices();

    HostBuffer<TriangleContainer4> tris;
    tris.reserve(idx.size() / 3);
    for (std::size_t t = 0; t < idx.size(); t += 3) {
        tris.push_back(make_tri(v[idx[t + 0]], v[idx[t + 1]], v[idx[t + 2]]));
    }
    return tris;
}

/** A view over the flat cube soup (no BVH; queries fall back to linear scan). */
TriangleMeshView
cube_view() {
    TriangleMeshView view {};
    view.vertices       = cube_vertices().data();
    view.indices        = cube_indices().data();
    view.triangle_count = static_cast<int>(cube_indices().size() / 3);
    return view;
}

}

/* ------------------------------------------------------------------ */
/* TriangleMesh: the host-side owner.                                  */
/* ------------------------------------------------------------------ */

TEST(TriangleMesh, DefaultMeshIsEmpty) {
    const TriangleMesh mesh;

    EXPECT_TRUE(mesh.triangles.empty());
    EXPECT_FALSE(mesh.is_valid());
}

TEST(TriangleMesh, DefaultMeshQueriesDegradeGracefully) {
    const TriangleMesh mesh;

    // Empty mesh: closest_point returns the query point unchanged and the
    // signed distance is +inf.
    const Float3 p(3.0f, -2.0f, 5.0f);
    const Float3 cp = mesh.closest_point(p);
    EXPECT_TRUE(cp == p);

    EXPECT_TRUE(std::isinf(mesh.signed_distance(p)));
    EXPECT_FALSE(mesh.is_inside(p, 0.0f));
}

TEST(TriangleMesh, ConstructFromTrianglesSetsCount) {
    const TriangleMesh mesh(cube_triangles());

    EXPECT_EQ(mesh.triangles.size(), std::size_t { 12 });
    EXPECT_TRUE(mesh.is_valid());
}

TEST(TriangleMesh, BoundEnclosesCube) {
    const TriangleMesh mesh(cube_triangles());
    const AABB box = mesh.bound();

    EXPECT_NEAR(box.lower_corner.x, 0.0f, kTol);
    EXPECT_NEAR(box.lower_corner.y, 0.0f, kTol);
    EXPECT_NEAR(box.lower_corner.z, 0.0f, kTol);
    EXPECT_NEAR(box.upper_corner.x, 1.0f, kTol);
    EXPECT_NEAR(box.upper_corner.y, 1.0f, kTol);
    EXPECT_NEAR(box.upper_corner.z, 1.0f, kTol);
}

TEST(TriangleMesh, CentroidIsCubeCenter) {
    const TriangleMesh mesh(cube_triangles());
    const Float3 c = mesh.centroid();

    EXPECT_NEAR(c.x, 0.5f, kTol);
    EXPECT_NEAR(c.y, 0.5f, kTol);
    EXPECT_NEAR(c.z, 0.5f, kTol);
}

TEST(TriangleMesh, SignedDistanceNegativeInside) {
    const TriangleMesh mesh(cube_triangles());

    // Center is 0.5 from every face; inside, so the signed distance is -0.5.
    EXPECT_NEAR(mesh.signed_distance(Float3(0.5f, 0.5f, 0.5f)), -0.5f, kTol);
}

TEST(TriangleMesh, SignedDistancePositiveOutside) {
    const TriangleMesh mesh(cube_triangles());

    // (0.5,0.5,2) is outside; nearest surface point (0.5,0.5,1) at distance 1.
    EXPECT_NEAR(mesh.signed_distance(Float3(0.5f, 0.5f, 2.0f)), 1.0f, kTol);
}

TEST(TriangleMesh, IsInsideClassifiesInteriorAndExterior) {
    const TriangleMesh mesh(cube_triangles());

    EXPECT_TRUE(mesh.is_inside(Float3(0.5f, 0.5f, 0.5f), 0.0f));
    EXPECT_FALSE(mesh.is_inside(Float3(2.0f, 2.0f, 2.0f), 0.0f));
}

TEST(TriangleMesh, ClosestPointLandsOnNearestFace) {
    const TriangleMesh mesh(cube_triangles());

    // Just below the top face: nearest surface point is (0.5,0.5,1).
    const Float3 cp = mesh.closest_point(Float3(0.5f, 0.5f, 0.9f));

    EXPECT_NEAR(cp.x, 0.5f, kTol);
    EXPECT_NEAR(cp.y, 0.5f, kTol);
    EXPECT_NEAR(cp.z, 1.0f, kTol);
}

TEST(TriangleMesh, TraceHitsCube) {
    const TriangleMesh mesh(cube_triangles());

    // Straight up through the bottom face, off the shared diagonal. The owning mesh has no
    // trace of its own; traversal lives on the view it hands out through a Geometry.
    const Ray ray(Float3(0.5f, 0.4f, -1.0f), Float3(0.0f, 0.0f, 1.0f));
    const HitSurface hit = mesh.make_device_geometry_view().trace(ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 1.0f, kTol);
    EXPECT_NEAR(hit.point.x, 0.5f, kTol);
    EXPECT_NEAR(hit.point.y, 0.4f, kTol);
    EXPECT_NEAR(hit.point.z, 0.0f, kTol);
    // Bottom face normal points along -z.
    EXPECT_NEAR(std::abs(hit.normal.z), 1.0f, kTol);
}

TEST(TriangleMesh, TraceMissesWhenAimedAway) {
    const TriangleMesh mesh(cube_triangles());

    const Ray ray(Float3(5.0f, 5.0f, 5.0f), Float3(0.0f, 0.0f, 1.0f));

    EXPECT_FALSE(mesh.make_device_geometry_view().trace(ray).is_intersecting);
}

TEST(TriangleMesh, MoveConstructionTransfersOwnership) {
    TriangleMesh source(cube_triangles());
    ASSERT_TRUE(source.is_valid());

    TriangleMesh moved(std::move(source));

    // The moved-to mesh owns the geometry and answers queries.
    EXPECT_EQ(moved.triangles.size(), std::size_t { 12 });
    EXPECT_TRUE(moved.is_valid());
    EXPECT_TRUE(moved.is_inside(Float3(0.5f, 0.5f, 0.5f), 0.0f));

    // The moved-from mesh is left empty and consistent.
    EXPECT_TRUE(source.triangles.empty());
    EXPECT_FALSE(source.is_valid());
}

TEST(TriangleMesh, MoveAssignmentTransfersOwnership) {
    TriangleMesh source(cube_triangles());
    TriangleMesh target;

    target = std::move(source);

    EXPECT_EQ(target.triangles.size(), std::size_t { 12 });
    EXPECT_TRUE(target.is_valid());
    EXPECT_NEAR(target.signed_distance(Float3(0.5f, 0.5f, 0.5f)), -0.5f, kTol);

    EXPECT_TRUE(source.triangles.empty());
    EXPECT_FALSE(source.is_valid());
}

/* ------------------------------------------------------------------ */
/* TriangleMeshView: the trivially-copyable query view.                */
/* ------------------------------------------------------------------ */

TEST(TriangleMeshView, DefaultViewIsEmpty) {
    const TriangleMeshView view {};

    EXPECT_EQ(view.triangle_count, 0);
    EXPECT_FALSE(view.is_valid());

    // Empty view degrades: closest_point returns the query point, signed
    // distance is +inf, and nothing is inside.
    const Float3 p(1.0f, 2.0f, 3.0f);
    EXPECT_TRUE(view.closest_point(p) == p);
    EXPECT_TRUE(std::isinf(view.signed_distance(p)));
    EXPECT_FALSE(view.is_inside(p, 0.0f));

    // Empty-view sentinels for the geometric accessors.
    EXPECT_FALSE(view.bound().is_valid());
    const Float3 c = view.centroid();
    EXPECT_NEAR(c.x, 0.0f, kTol);
    EXPECT_NEAR(c.y, 0.0f, kTol);
    EXPECT_NEAR(c.z, 0.0f, kTol);
}

TEST(TriangleMeshView, TriangleCountAndValidity) {
    const TriangleMeshView view = cube_view();

    EXPECT_EQ(view.triangle_count, 12);
    EXPECT_TRUE(view.is_valid());
}

TEST(TriangleMeshView, IsValidFalseWithDegenerateTriangle) {
    // A single collinear triangle: valid pointers but zero area.
    const std::vector<Float3> verts = {
        Float3(0.0f, 0.0f, 0.0f),
        Float3(1.0f, 0.0f, 0.0f),
        Float3(2.0f, 0.0f, 0.0f),
    };
    const std::vector<int> idx = { 0, 1, 2 };

    TriangleMeshView view {};
    view.vertices       = verts.data();
    view.indices        = idx.data();
    view.triangle_count = 1;

    EXPECT_FALSE(view.is_valid());
}

TEST(TriangleMeshView, BoundEnclosesCube) {
    const AABB box = cube_view().bound();

    EXPECT_NEAR(box.lower_corner.x, 0.0f, kTol);
    EXPECT_NEAR(box.lower_corner.y, 0.0f, kTol);
    EXPECT_NEAR(box.lower_corner.z, 0.0f, kTol);
    EXPECT_NEAR(box.upper_corner.x, 1.0f, kTol);
    EXPECT_NEAR(box.upper_corner.y, 1.0f, kTol);
    EXPECT_NEAR(box.upper_corner.z, 1.0f, kTol);
}

TEST(TriangleMeshView, CentroidIsCubeCenter) {
    const Float3 c = cube_view().centroid();

    EXPECT_NEAR(c.x, 0.5f, kTol);
    EXPECT_NEAR(c.y, 0.5f, kTol);
    EXPECT_NEAR(c.z, 0.5f, kTol);
}

TEST(TriangleMeshView, ClosestPointOnNearestFace) {
    const TriangleMeshView view = cube_view();

    const Float3 cp = view.closest_point(Float3(0.5f, 0.5f, 0.9f));

    EXPECT_NEAR(cp.x, 0.5f, kTol);
    EXPECT_NEAR(cp.y, 0.5f, kTol);
    EXPECT_NEAR(cp.z, 1.0f, kTol);
}

TEST(TriangleMeshView, ClosestNormalNearBottomFace) {
    const TriangleMeshView view = cube_view();

    // Point closest to the bottom (z=0) face; its geometric normal is -z.
    const Float3 n = view.closest_normal(Float3(0.3f, 0.2f, 0.1f));

    EXPECT_NEAR(std::abs(n.z), 1.0f, kTol);
    EXPECT_LT(n.z, 0.0f);
}

TEST(TriangleMeshView, SignedDistanceNegativeInside) {
    const TriangleMeshView view = cube_view();

    EXPECT_NEAR(view.signed_distance(Float3(0.5f, 0.5f, 0.5f)), -0.5f, kTol);
}

TEST(TriangleMeshView, SignedDistancePositiveOutside) {
    const TriangleMeshView view = cube_view();

    EXPECT_NEAR(view.signed_distance(Float3(0.5f, 0.5f, 2.0f)), 1.0f, kTol);
}

TEST(TriangleMeshView, IsInsideClassifiesInteriorAndExterior) {
    const TriangleMeshView view = cube_view();

    EXPECT_TRUE(view.is_inside(Float3(0.5f, 0.5f, 0.5f), 0.0f));
    EXPECT_FALSE(view.is_inside(Float3(2.0f, 2.0f, 2.0f), 0.0f));

    // Erosion: an interior point close to the surface drops out with a
    // negative tolerance whose magnitude exceeds its distance to the surface.
    EXPECT_FALSE(view.is_inside(Float3(0.5f, 0.5f, 0.95f), -0.1f));
}

TEST(TriangleMeshView, IsOnSurfaceProximity) {
    const TriangleMeshView view = cube_view();

    // On the bottom face.
    EXPECT_TRUE(view.is_on_surface(Float3(0.5f, 0.5f, 0.0f), 0.0f));

    // 0.1 inside: outside a zero shell, inside a 0.2 shell.
    EXPECT_FALSE(view.is_on_surface(Float3(0.5f, 0.5f, 0.1f), 0.0f));
    EXPECT_TRUE(view.is_on_surface(Float3(0.5f, 0.5f, 0.1f), 0.2f));

    // Negative tolerance is rejected as invalid.
    EXPECT_FALSE(view.is_on_surface(Float3(0.5f, 0.5f, 0.0f), -0.1f));
}

TEST(TriangleMeshView, WindingNumberInsideVsOutside) {
    const TriangleMeshView view = cube_view();

    // ~+/-1 inside a closed oriented surface, ~0 outside.
    EXPECT_NEAR(std::abs(view.winding_number(Float3(0.5f, 0.5f, 0.5f))), 1.0f, 1e-2f);
    EXPECT_NEAR(view.winding_number(Float3(5.0f, 5.0f, 5.0f)), 0.0f, 1e-2f);
}

TEST(TriangleMeshView, TraceWithoutABuiltBvhReportsNoHit) {
    // A hand-built view carries the flat vertex/index arrays but no BVH. Unlike the other
    // queries, trace() traverses the hierarchy on a CPU build (the flat-soup fallback is
    // compiled in only for the host side of a CUDA build), so it finds nothing here.
    const TriangleMeshView view = cube_view();

    const Ray ray(Float3(0.5f, 0.4f, -1.0f), Float3(0.0f, 0.0f, 1.0f));

    EXPECT_FALSE(view.trace(ray).is_intersecting);
}

TEST(TriangleMeshView, TraceHitsAndMissesOnAMeshBuiltView) {
    const TriangleMesh mesh(cube_triangles());
    const Geometry geometry = mesh.make_device_geometry_view();

    ASSERT_EQ(geometry.type, GeometryType::triangle_mesh);

    const TriangleMeshView view = geometry.triangle_mesh;

    const Ray hit_ray(Float3(0.5f, 0.4f, -1.0f), Float3(0.0f, 0.0f, 1.0f));
    const HitSurface hit = view.trace(hit_ray);

    ASSERT_TRUE(hit.is_intersecting);
    EXPECT_NEAR(hit.distance, 1.0f, kTol);
    EXPECT_NEAR(hit.point.z, 0.0f, kTol);

    const Ray miss_ray(Float3(5.0f, 5.0f, 5.0f), Float3(0.0f, 0.0f, 1.0f));
    EXPECT_FALSE(view.trace(miss_ray).is_intersecting);
}
